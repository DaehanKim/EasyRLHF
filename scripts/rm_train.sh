#!/bin/bash
WANDB_PROJECT="easy-rlhf" \
TOKENIZERS_PARALLELISM="false" \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed --master_port $1 --module EasyRLHF.reward_model \
--model_name_or_path "gpt2-xl" \
--output_dir "outputs/rm-gpt2xl-bsz32-filter1024-eos-data_all-bcewlogitsloss-resume" \
--train_file "data/helpful-base/train.jsonl,data/helpful-online/train.jsonl,data/helpful-rejection-sampled/train.jsonl" \
--valid_file "data/helpful-base/test.jsonl,data/helpful-online/test.jsonl,data/helpful-rejection-sampled/test.jsonl" \
--do_train 1 \
--overwrite_output_dir \
--evaluation_strategy steps \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--fp16 1 \
--num_train_epochs 10 \
--eval_steps 1000 \
--save_steps 1000 \
--save_strategy steps \
--save_total_limit 3 \
--logging_steps 20 \
--max_seq_length 1024 \
--report_to wandb \
--run_name rm-gpt2xl-bsz32-filter1024-eos-data_all-bcewlogitsloss \
--learning_rate 1e-5 \
--deepspeed configs/ds_config.json \
--resume /workspace/EasyRLHF/outputs/rm-gpt2xl-bsz32-filter1024-eos-data_all-bcewlogitsloss/checkpoint-5500