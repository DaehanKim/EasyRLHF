#!/bin/bash
cd ../EasyRLHF

WANDB_PROJECT="easy-rlhf" \
TOKENIZERS_PARALLELISM="false" \
PJRT_SELECT_DEFAULT_DEVICE=0 \
PJRT_DEVICE=TPU \
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=27374182400 \
python xla_spawn.py --num_cores=8 sft_train.py \
--model_name "gpt2-medium" \
--dataset_name "open-orca/slimorca-dedup" \
--output_dir "../outputs/gpt2m-sft-slimopcadedup" \
--evaluation_strategy steps \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--bf16 1 \
--num_train_epochs 3 \
--eval_steps 1000 \
--save_steps 2000 \
--save_strategy steps \
--save_total_limit 3 \
--logging_steps 1 \
--max_seq_length 1024 \
--report_to wandb \
--run_name gpt2m-sft-slimopcadedup \
--learning_rate 1e-5 \
--deepspeed ../configs/tpu_ds_config.json \
--neftune_noise_alpha 5.0 \
--gradient_checkpointing 1 