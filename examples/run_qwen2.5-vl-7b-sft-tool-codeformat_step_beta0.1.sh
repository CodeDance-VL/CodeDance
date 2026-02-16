# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1

#export NCCL_DEBUG=INFO
#ulimit -n 65535
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export NCCL_DEBUG=INFO

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

mkdir -p logs

rollout_name="sglang" # sglang or vllm
PT_CKPT_PATH=YOUR_SFT_MODEL


df_toolbox_v1_path=${DATA_ROOT}/df_toolbox_v1_new_format.parquet
df_toolbox_v2_path=${DATA_ROOT}/df_toolbox_v2_new_format.parquet
df_thinklite_new_path=${DATA_ROOT}/df_thinklite_new_format.parquet

chart_rl_path=${DATA_ROOT}/chart_refocus_train_filter_v1.parquet
pixmo_train_path=${DATA_ROOT}/pixmo_train_filter.parquet
sa1b_train_path=${DATA_ROOT}/sa1b_rl_filtered.parquet

v_star_path=${DATA_ROOT}/vstar_val.parquet
mathvision_path=${DATA_ROOT}/mathvision_testmini.parquet
mathvista_path=${DATA_ROOT}/mathvista_testmini.parquet
mathverse_path=${DATA_ROOT}/mathverse_vision_only_testmini.parquet


pixmo_test_path=${DATA_ROOT}/pixmo_test.parquet
count_test_path=${DATA_ROOT}/countbench_QA_eval.parquet

train_files="['$df_toolbox_v1_path','$df_toolbox_v2_path', '$df_thinklite_new_path', '$chart_rl_path', '$pixmo_train_path', '$sa1b_train_path']"
val_files="['$v_star_path', '$count_test_path', '$pixmo_test_path','$mathvision_path','$mathvista_path','$mathverse_path']"

nohup python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multiturn_config.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256\
    data.max_prompt_length=10240 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$PT_CKPT_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4\
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    ++algorithm.use_step_reward=True\
    ++algorithm.tool_decay_gamma=4\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='CodeDance' \
    trainer.experiment_name='codeformat-34k-step-beta-0.2' \
    trainer.n_gpus_per_node=8\
    trainer.nnodes=2 \
    actor_rollout_ref.nccl_timeout=3600 \
    trainer.save_freq=20 \
    trainer.test_freq=300 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir='./rollout_sft_step/'\
    trainer.validation_data_dir='./validation_sft_step/'\
    custom_reward_function.path=./verl/utils/reward_score/llm_judge_qwen_tool_step.py \
    custom_reward_function.name=compute_score_batch \
    reward_model.reward_manager=batch \
    data.train_files="$train_files"   \
    data.val_files="$val_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config.yaml" \
    "$@" >> logs/run_$(date +%F_%H-%M-%S)_release_test.log 2>&1 & echo $! > logs/run.pid