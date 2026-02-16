# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import os
import logging
import difflib
import torch
from pydantic import BaseModel, PrivateAttr, ConfigDict, model_validator

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema, ToolResponse
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.vision_utils import process_image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY_FOR_TEMPLATE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class TextContent(BaseModel):
    """Text content for multimodal messages."""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content for multimodal messages."""
    type: Literal["image", "image_url"] = "image_url"
    image_url: Optional[Union[str, Dict[str, Any]]] = None  # For image_url type
    image: Optional[Union[str, bytes, Any]] = None  # For image type (direct image data)

class VideoContent(BaseModel):
    """Video content for multimodal messages."""
    type: Literal["video"] = "video"
    video: Optional[Union[str, bytes, Any]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent, VideoContent]]]
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None
     

    def to_chatml_format(self) -> Dict[str, Any]:
        """Convert to ChatML format for compatibility."""
        result = {"role": self.role}
        
        if isinstance(self.content, str):
            result["content"] = self.content
        elif isinstance(self.content, list):
            # Convert to OpenAI-style multimodal format
            content_list = []
            for item in self.content:
                if isinstance(item, TextContent):
                    content_list.append({"type": "text", "text": item.text})
                elif isinstance(item, ImageContent):
                    if item.type == "image_url":
                        if isinstance(item.image_url, str):
                            content_list.append({"type": "image_url", "image_url": {"url": item.image_url}})
                        elif isinstance(item.image_url, dict):
                            content_list.append({"type": "image_url", "image_url": item.image_url})
                        else:
                            # Fallback to placeholder if no URL provided
                            content_list.append({"type": "image_url"})
                    elif item.type == "image":
                        # For direct image data, include placeholder; actual image is passed via processor inputs
                        content_list.append({"type": "image"})
                elif isinstance(item, VideoContent):
                    # For direct video data, include placeholder; actual video is passed via processor inputs
                    content_list.append({"type": "video"})
            result["content"] = content_list
        
        if self.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        
        return result
    
class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"


class TokenizationSanityCheckModeEnum(str, Enum):
    """The enum for tokenization sanity check mode."""

    DISABLE = "disable"
    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"

class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _base_template_len: Optional[int] = PrivateAttr(default=None)
    _base_template_len_with_gen: Optional[int] = PrivateAttr(default=None)
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    multi_modal_keys: Optional[List[str]] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    multi_modal_inputs: Optional[Dict[str, torch.Tensor]] = None
    tool_schemas: Optional[list[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    interaction_kwargs: Dict[str, Any] = {}
    input_ids: Optional[torch.Tensor] = None
    prompt_ids: Optional[torch.Tensor] = None
    response_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
    response_attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    prompt_position_ids: Optional[torch.Tensor] = None
    response_position_ids: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    prompt_loss_mask: Optional[torch.Tensor] = None
    response_loss_mask: Optional[torch.Tensor] = None
    reward_scores: Dict[str, List[float]]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}
    output_token_ids: torch.Tensor | None = None
    rollout_log_probs: torch.Tensor | None = None

    use_inference_chat_template: bool = True
    tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum = TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE
    generation_prompt_ids: Optional[torch.Tensor] = None
    base_conv_wo_gen_prompt_end_pos: int = 0
    base_conv_with_gen_prompt_end_pos: int = 0
    num_turns: int = 0

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        messages = values.get("messages")
        if not messages:
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        processing_class = values.pop("processing_class", None)
        if processing_class is None:
            raise ValueError("processing_class is required for AsyncRolloutRequest initialization")
        max_prompt_len = values.get("max_prompt_len")
        if not max_prompt_len:
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        # normalize messages to pydantic model with multimodal content list
        normalized_messages = []
        for m in messages:
            if isinstance(m, Message):
                # Already a Message; ensure content is a list
                role = m.role
                content = m.content
                tool_calls = m.tool_calls
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                tool_calls = m.get("tool_calls")
            else:
                # Fallback: treat as user text message
                role = "user"
                content = str(m)
                tool_calls = None

            content_items = []
            if isinstance(content, str):
                content_items = [TextContent(text=content)]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, TextContent) or isinstance(item, ImageContent) or isinstance(item, VideoContent):
                        content_items.append(item)
                    elif isinstance(item, dict):
                        t = item.get("type")
                        if t == "text":
                            content_items.append(TextContent(text=item.get("text", "")))
                        elif t == "image_url":
                            content_items.append(ImageContent(type="image_url", image_url=item.get("image_url")))
                        elif t == "image":
                            content_items.append(ImageContent(type="image", image=item.get("image")))
                        elif t == "video":
                            content_items.append(VideoContent(type="video", video=item.get("video")))
                    else:
                        # Ignore unsupported item types
                        continue
            else:
                content_items = [TextContent(text=str(content) if content is not None else "")] 

            new_m = {"role": role, "content": content_items}
            if tool_calls is not None:
                new_m["tool_calls"] = tool_calls
            normalized_messages.append(Message.model_validate(new_m))

        values["messages"] = normalized_messages
        # init multimodal containers
        if not values.get("multi_modal_keys"):
            values["multi_modal_keys"] = ["image", "video"]
        if not values.get("multi_modal_data"):
            values["multi_modal_data"] = {k: [] for k in values["multi_modal_keys"]}
        else:
            for k in values["multi_modal_keys"]:
                values["multi_modal_data"].setdefault(k, [])
        if not values.get("multi_modal_inputs"):
            values["multi_modal_inputs"] = {}

        tools = [tool.model_dump() for tool in values.get("tool_schemas", [])] or None

        # Build base lengths for template slicing
        base_ids_wo_gen = cls._handle_apply_chat_template(
            processing_class,
            values["messages"],
            multi_modal_data=values["multi_modal_data"],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        # Compute base conversation template ids and adjust to exclude trailing newline(s)
        base_conv_wo_gen_ids = cls._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY_FOR_TEMPLATE,
            multi_modal_data=values["multi_modal_data"],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        values["base_conv_wo_gen_prompt_end_pos"] = base_conv_wo_gen_ids.shape[-1]

        base_conv_with_gen_ids = cls._handle_apply_chat_template(
            processing_class,
            BASE_CHAT_HISTORY_FOR_TEMPLATE,
            multi_modal_data=values["multi_modal_data"],
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        values["base_conv_with_gen_prompt_end_pos"] = base_conv_with_gen_ids.shape[-1]

        # If no pre-tokenized tensors provided, compute them
        if (
            values.get("input_ids") is None
            or values.get("attention_mask") is None
            or values.get("position_ids") is None
        ):
            tokenization_dict_with_prompt = cls._handle_apply_chat_template(
                processing_class,
                values["messages"],
                multi_modal_data=values["multi_modal_data"],
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
            values["input_ids"] = tokenization_dict_with_prompt["input_ids"]
            values["attention_mask"] = tokenization_dict_with_prompt["attention_mask"]
            # preserve multi_modal_inputs besides ids/mask
            multi_modal_inputs = dict(tokenization_dict_with_prompt)
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            values["multi_modal_inputs"] = multi_modal_inputs
            # compute position_ids
            values["position_ids"] = values["prompt_position_ids"] = cls._get_position_ids(
                processing_class,
                values["input_ids"],
                values["attention_mask"],
                multi_modal_inputs,
            )
            if values["input_ids"].shape[-1] > max_prompt_len:
                logger.warning(
                    f"Prompt length {values['input_ids'].shape[-1]} exceeds max_prompt_len {max_prompt_len}."
                )
        values["prompt_ids"] = values["input_ids"]
        values["prompt_attention_mask"] = values["attention_mask"]
        values["loss_mask"] = values["prompt_loss_mask"] = torch.zeros_like(values["input_ids"], dtype=torch.bool)
        values["generation_prompt_ids"] = values["input_ids"][..., base_ids_wo_gen.shape[-1]:]
        return values

    @staticmethod
    def _handle_apply_chat_template(
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        messages: List[Union[Message, Dict[str, Any]]],
        multi_modal_data: Dict[str, Any],
        tools: Optional[list[OpenAIFunctionToolSchema]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_dict: bool = False,
    ):
        # Convert Message objects to ChatML format to ensure proper multimodal placeholders
        raw_messages = [
            m.to_chatml_format() if isinstance(m, Message) else m for m in messages
        ]
        raw_prompt = processing_class.apply_chat_template(
            raw_messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        if not tokenize:
            return raw_prompt
        if isinstance(processing_class, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            if any(len(v) > 0 for v in multi_modal_data.values()):
                logger.warning("There is multi_modal_data but you are not using a processor. It will be ignored.")
            model_inputs = processing_class(text=[raw_prompt], return_tensors="pt")
        elif isinstance(processing_class, ProcessorMixin):
            images = images if len(images := multi_modal_data.get("image", [])) > 0 else None
            videos = videos if len(videos := multi_modal_data.get("video", [])) > 0 else None
            model_inputs = processing_class(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported processing class type: {type(processing_class)}")
        model_inputs = dict(model_inputs)
        return model_inputs if return_dict else model_inputs["input_ids"]
    
    @staticmethod
    def _get_position_ids(
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # special case for qwen2vl
        is_qwen2vl = (
            hasattr(processing_class, "image_processor")
            and "Qwen2VLImageProcessor" in processing_class.image_processor.__class__.__name__
        )
        if is_qwen2vl:
            from verl.models.transformers.qwen2_vl import get_rope_index

            image_grid_thw = video_grid_thw = second_per_grid_ts = None
            if multi_modal_inputs:
                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"input_ids should be 2D with batch size 1, but got shape {input_ids.shape}"
            )
            assert attention_mask.dim() == 2 and attention_mask.shape[0] == 1, (
                f"attention_mask should be 2D with batch size 1, but got shape {attention_mask.shape}"
            )
            new_position_ids = get_rope_index(
                processing_class,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            )
            return new_position_ids  # (3, seq_len)
        else:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)


    def _update_multi_modal_inputs(self, new_multi_modal_inputs: dict[str, torch.Tensor]) -> None:
        """
        Update the multi_modal_inputs of the request in additive manner.
        """
        for key in new_multi_modal_inputs:
            input_tensor = new_multi_modal_inputs[key]
            self.multi_modal_inputs[key] = (
                torch.cat([self.multi_modal_inputs[key], input_tensor], dim=0)
                if key in self.multi_modal_inputs
                else input_tensor
            )

    def _update_input_ids(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        new_input_ids: torch.Tensor,
        attention_mask: bool,
        loss_mask: bool,
        new_multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=-1)
        attention_mask = torch.ones_like(new_input_ids) * int(attention_mask)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=-1)
        loss_mask = torch.ones_like(new_input_ids) * int(loss_mask)
        self.loss_mask = torch.cat([self.loss_mask, loss_mask], dim=-1)

        if new_multi_modal_inputs:
            self._update_multi_modal_inputs(new_multi_modal_inputs)

        new_position_ids = self._get_position_ids(
            processing_class, new_input_ids, attention_mask, new_multi_modal_inputs
        )

        last_pos = self.position_ids[..., -1:]
        new_position_ids = new_position_ids + (last_pos + 1)

        self.position_ids = torch.cat([self.position_ids, new_position_ids], dim=-1)

        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"""Request {self.request_id} has different length of {self.input_ids.shape[-1]=}, 
            {self.attention_mask.shape[-1]=}, {self.position_ids.shape[-1]=}, {self.loss_mask.shape[-1]=}"""

    def _remove_generation_prompt_ids_if_present(self) -> None:
        if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all():
            self.input_ids = self.input_ids[..., : -self.generation_prompt_ids.shape[-1]]
            self.attention_mask = self.attention_mask[..., : -self.generation_prompt_ids.shape[-1]]
            self.position_ids = self.position_ids[..., : -self.generation_prompt_ids.shape[-1]]
            self.loss_mask = self.loss_mask[..., : -self.generation_prompt_ids.shape[-1]]

    def get_generation_prompt_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> list[int]:
        gen_ids = (
            None
            if self.input_ids[..., -self.generation_prompt_ids.shape[-1] :].eq(self.generation_prompt_ids).all()
            else self.generation_prompt_ids
        )
        if gen_ids is not None:
            self._update_input_ids(processing_class, gen_ids, attention_mask=True, loss_mask=False)
        if self.use_inference_chat_template:
            messages = self.messages
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
            return ids.squeeze(0).tolist()
        else:
            return self.input_ids.squeeze(0).tolist()

    def add_user_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        messages = [*BASE_CHAT_HISTORY_FOR_TEMPLATE, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        content_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data={},
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )[..., self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=False)


    def add_assistant_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
        content_ids: Optional[torch.Tensor] = None,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        if content_ids is None:
            messages = [*BASE_CHAT_HISTORY_FOR_TEMPLATE, self.messages[-1]]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            content_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data={},
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )[..., self.base_conv_with_gen_prompt_end_pos :]

        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=True)


    def add_tool_response_messages(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        contents: List[Any],
    ) -> bool:
        if not contents:
            return False
        original_messages = list(self.messages)
        delta_multi_modal_data = {key: [] for key in self.multi_modal_keys}
        for c in contents:
            if isinstance(c, ToolResponse):
                if c.is_text_only():
                    self.messages.append(Message(role="tool", content=c.text))
                else:
                    content_items: List[Union[TextContent, ImageContent, VideoContent]] = []
                    if c.image:
                        for img in c.image:
                            processed = process_image({"image": img})
                            content_items.append(ImageContent(type="image", image=processed))
                            delta_multi_modal_data["image"].append(processed)
                    if c.video:
                        for vid in c.video:
                            content_items.append(VideoContent(type="video", video=vid))
                        delta_multi_modal_data["video"].extend(c.video)
                    if c.text:
                        content_items.append(TextContent(text=c.text))
                    self.messages.append(Message(role="tool", content=content_items))
            elif isinstance(c, dict):
                content_items: List[Union[TextContent, ImageContent, VideoContent]] = []
                for item in c.get("content", []):
                    if item.get("type") == "text" and "text" in item:
                        content_items.append(TextContent(text=item["text"]))
                    elif item.get("type") == "image_url" and "image_url" in item:
                        content_items.append(ImageContent(type="image_url", image_url=item["image_url"]))
                    elif item.get("type") == "image" and "image" in item:
                        # breakpoint()
                        processed = process_image({"image": item["image"]})
                        content_items.append(ImageContent(type="image", image=processed))
                        delta_multi_modal_data["image"].append(processed)
                    elif item.get("type") == "video" and "video" in item:
                        content_items.append(VideoContent(type="video", video=item["video"]))
                        delta_multi_modal_data["video"].append(item["video"]) 
                self.messages.append(Message(role="tool", content=content_items))
            else:
                self.messages.append(Message(role="tool", content=str(c)))
        # Avoid duplication: generation prompt tokens may have been appended prior to tool call
        # Remove them before adding tool response chunks to keep template tokens aligned
        self._remove_generation_prompt_ids_if_present()
        messages = [*BASE_CHAT_HISTORY_FOR_TEMPLATE, *self.messages[-len(contents) :]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        content_info = self._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=delta_multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )
        content_ids = content_info["input_ids"][..., self.base_conv_wo_gen_prompt_end_pos :]
        current_response_len = self.input_ids[..., self.prompt_ids.shape[-1] :].shape[-1]
        if current_response_len + content_ids.shape[-1] > self.max_response_len:
            self.messages = original_messages
            return False
        for key in self.multi_modal_keys:
            if len(delta_multi_modal_data[key]) > 0:
                self.multi_modal_data[key].extend(delta_multi_modal_data[key])
        multi_modal_inputs = dict(content_info)
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)
        self._update_input_ids(
            processing_class,
            content_ids,
            attention_mask=True,
            loss_mask=False,
            new_multi_modal_inputs=multi_modal_inputs,
        )
        return True

    def _get_prompt_diffs(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        full_prompt_ids: torch.Tensor,
        current_prompt_ids: torch.Tensor,
        diff_surrounding_chars: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get differences between full prompt and current prompt with surrounding context.

        This helps debug tokenization mismatches by showing differing chunks with
        additional characters before and after to locate issues in the chat template.
        """
        full_prompt_ids = full_prompt_ids.squeeze(0)
        current_prompt_ids = current_prompt_ids.squeeze(0)
        full_prompt = processing_class.decode(full_prompt_ids, skip_special_tokens=False)
        current_prompt = processing_class.decode(current_prompt_ids, skip_special_tokens=False)
        s = difflib.SequenceMatcher(None, full_prompt, current_prompt, autojunk=False)
        diffs: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                continue
            start_i = max(0, i1 - diff_surrounding_chars)
            end_i = min(len(full_prompt), i2 + diff_surrounding_chars)
            start_j = max(0, j1 - diff_surrounding_chars)
            end_j = min(len(current_prompt), j2 + diff_surrounding_chars)
            diffs.append(
                {
                    "full_prompt_chunk": full_prompt[start_i:end_i],
                    "current_prompt_chunk": current_prompt[start_j:end_j],
                    "indices": (start_i, end_i, start_j, end_j),
                }
            )
        return diffs

    def finalize(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        reward_scores: Dict[str, List[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        # In case we failed to generate the assistant message and the generation prompt ids were already added to
        # input_ids, remove them from the end of input_ids
        # removed stray breakpoint to avoid halting during rollout
        # breakpoint()
        # In case we failed to generate the assistant message and the generation prompt ids were already added to
        # input_ids, remove them from the end of input_ids
        self._remove_generation_prompt_ids_if_present()
        self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :]
        # # print all masked tokens
        # if "<tool_response>" in processing_class.tokenizer.decode(self.input_ids[0][self.attention_mask[0] == 1]):
        #     print("********************[Masked tokens]**********************")
        #     print(processing_class.tokenizer.decode(self.input_ids[0][self.loss_mask[0] == 0]))
        #     # print all non-masked tokens
        #     print("********************[Non-masked tokens]**********************")
        #     print(processing_class.tokenizer.decode(self.input_ids[0][self.loss_mask[0] == 1]))
        #     exit(0)

        # if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.DISABLE:
        #     diff_surrounding_chars = 10
        #     messages = [m.to_chatml_format() for m in self.messages]
        #     tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        #     full_prompt_info = self._handle_apply_chat_template(
        #         processing_class,
        #         messages,
        #         multi_modal_data=self.multi_modal_data,
        #         tools=tools,
        #         add_generation_prompt=False,
        #         tokenize=True,
        #         return_dict=True,
        #     )
        #     full_prompt_ids = full_prompt_info["input_ids"]

        #     full_prompt_multi_modal_inputs = full_prompt_info.copy()
        #     full_prompt_multi_modal_inputs.pop("input_ids", None)
        #     full_prompt_multi_modal_inputs.pop("attention_mask", None)

        #     for multi_modal_inputs_key in self.multi_modal_inputs:
        #         if multi_modal_inputs_key in full_prompt_multi_modal_inputs:
        #             if (
        #                 not self.multi_modal_inputs[multi_modal_inputs_key]
        #                 .eq(full_prompt_multi_modal_inputs[multi_modal_inputs_key])
        #                 .all()
        #             ):
        #                 logger.warning(
        #                     f"Multi-modal data {multi_modal_inputs_key} is not consistent. "
        #                     f"This may lead to unexpected behavior during training. "
        #                     f"Please review your multi_modal_inputs logic."
        #                 )
        #         else:
        #             logger.warning(
        #                 f"Multi-modal inputs key {multi_modal_inputs_key} is not found in the multi_modal_inputs. "
        #                 f"This may lead to unexpected behavior during training."
        #                 f"Please review your multi_modal_inputs logic."
        #             )

        #     diffs = self._get_prompt_diffs(
        #         processing_class, full_prompt_ids, self.input_ids, diff_surrounding_chars=diff_surrounding_chars
        #     )
        #     if diffs:
        #         log_warning = False
        #         if self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.STRICT:
        #             log_warning = False # TODO: True
        #         elif self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE:
        #             non_strippable_diffs_exist = any(
        #                 d["full_prompt_chunk"].strip() or d["current_prompt_chunk"].strip() for d in diffs
        #             )
        #             if non_strippable_diffs_exist:
        #                 log_warning = True

        #         if log_warning:
        #             mode_str = f" ({self.tokenization_sanity_check_mode.value})"
        #             logger.warning(
        #                 f"Inconsistent training and inference tokenization detected{mode_str}. This may lead to "
        #                 f"unexpected behavior during training. Please review your chat template to determine if this "
        #                 f"is intentional. For more information, refer to the multiturn README.md."
        #             )
        #             logger.warning(
        #                 f"Showing {diff_surrounding_chars} characters before and after the diffs for context and "
        #                 f"better readability."
        #             )
        #             diff_details_list = []
        #             for d in diffs:
        #                 i1, i2, j1, j2 = d["indices"]
        #                 diff_details_list.append(
        #                     f"idx {i1}:{i2} -> {j1}:{j2} | full_prompt_chunk: {repr(d['full_prompt_chunk'])} | "
        #                     f"current_prompt_chunk: {repr(d['current_prompt_chunk'])}"
        #                 )
        #             diff_details = "\n".join(diff_details_list)
        #             logger.warning(f"Found differences:\n{diff_details}")

        # # position_ids consistency check
        # expected_position_ids = self._get_position_ids(
        #     processing_class,
        #     full_prompt_ids,
        #     full_prompt_info["attention_mask"],
        #     full_prompt_multi_modal_inputs,
        # )
        # # Compare expected vs current position_ids
        # if expected_position_ids.shape != self.position_ids.shape or not expected_position_ids.eq(self.position_ids).all().item():
        #     mode_str = f" ({self.tokenization_sanity_check_mode.value})"
        #     logger.warning(
        #         f"Position IDs mismatch detected{mode_str}. This may lead to unexpected behavior during training."
        #     )
        #     logger.warning(
        #         f"Expected position_ids shape {expected_position_ids.shape}, current shape {self.position_ids.shape}."
        #     )
        #     if expected_position_ids.shape == self.position_ids.shape:
        #         mismatch_mask = ~expected_position_ids.eq(self.position_ids)
        #         mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
        #         sample_indices = mismatch_indices[:10].tolist()
        #         logger.warning(f"First mismatched indices (up to 10): {sample_indices}")

        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")

        self.truncate_output_ids(processing_class)
        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"Request {self.request_id} tensor length mismatch after finalize"

    def truncate_output_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> None:
        self.input_ids = self.input_ids[..., : self.max_model_len]
        self.attention_mask = self.attention_mask[..., : self.max_model_len]
        self.position_ids = self.position_ids[..., : self.max_model_len]
        self.loss_mask = self.loss_mask[..., : self.max_model_len]
        self.response_ids = self.input_ids[..., self.prompt_ids.shape[-1] :][..., : self.max_response_len]
        self.response_attention_mask = self.attention_mask[..., self.prompt_attention_mask.shape[-1] :][
            ..., : self.max_response_len
        ]
        self.response_position_ids = self.position_ids[..., self.prompt_position_ids.shape[-1] :][
            ..., : self.max_response_len
        ]
        self.response_loss_mask = self.loss_mask[..., self.prompt_loss_mask.shape[-1] :][..., : self.max_response_len]

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)
