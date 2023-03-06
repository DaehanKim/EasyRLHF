import subprocess
import argparse
import sys


def rm_train():
    parser = argparse.ArgumentParser(description="Script for training a reward model")
    parser.add_argument("--devices", default="0", help="gpu ids to train a model on")
    parser.add_argument(
        "--master_port", default="50000", help="master port for deepspeed"
    )
    parser.add_argument(
        "--project_name", default="easy-rlhf", help="project name for wandb tracking"
    )
    parser.add_argument(
        "--output_dir", default="outputs/my-reward-model", help="output directory"
    )
    parser.add_argument(
        "--train_file",
        default="data/helpful-base/train.jsonl,data/helpful-online/train.jsonl,data/helpful-rejection-sampled/train.jsonl",
        help="output directory",
    )
    parser.add_argument(
        "--valid_file",
        default="data/helpful-base/test.jsonl,data/helpful-online/test.jsonl,data/helpful-rejection-sampled/test.jsonl",
        help="output directory",
    )

    args = parser.parse_args()

    subprocess.run(
        f"""WANDB_PROJECT={args.project_name} \
TOKENIZERS_PARALLELISM="false" \
CUDA_VISIBLE_DEVICES={args.devices} \
deepspeed --master_port {args.master_port} --module EasyRLHF.reward_model \
--model_name_or_path "gpt2-xl" \
--output_dir {args.output_dir} \
--train_file {args.train_file} \
--valid_file {args.valid_file} \
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
--warmup_steps 1000 \
--report_to "wandb" \
--run_name {args.output_dir.split("/")[-1]} \
--learning_rate 1e-5 \
--deepspeed configs/ds_config.json""",
        shell=True,
    )
