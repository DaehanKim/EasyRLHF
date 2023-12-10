#!/bin/bash
cd ../EasyRLHF

WANDB_PROJECT="easy-rlhf" \
TOKENIZERS_PARALLELISM="false" \
PJRT_SELECT_DEFAULT_DEVICE=0 \
PJRT_DEVICE=TPU \
python xla_spawn.py reward_model.py \
--model_name_or_path "gpt2-large" \
--output_dir "../outputs/rm-gpt2l-bsz8-filter1024-eos-data_all-bcewlogitsloss-resume" \
--train_file "../data/helpful-base/train.jsonl,../data/helpful-online/train.jsonl,../data/helpful-rejection-sampled/train.jsonl" \
--valid_file "../data/helpful-base/test.jsonl,../data/helpful-online/test.jsonl,../data/helpful-rejection-sampled/test.jsonl" \
--do_train 1 \
--overwrite_output_dir \
--evaluation_strategy steps \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--bf16 1 \
--num_train_epochs 10 \
--eval_steps 1000 \
--save_steps 1000 \
--save_strategy steps \
--save_total_limit 3 \
--logging_steps 1 \
--max_seq_length 1024 \
--report_to wandb \
--run_name rm-gpt2l-bsz8-filter1024-eos-data_all-bcewlogitsloss-resume \
--learning_rate 1e-5 \
--deepspeed "../configs/ds_config.json"