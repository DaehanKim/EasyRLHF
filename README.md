<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# EasyRLHF
EasyRLHF aims to provide an easy and minimal interface to train aligned language models, using off-the-shelf solutions and datasets (i.e. HF Trainer, HF Datasets, Deepspeed, trl).

Following sections will cover rough concepts of alignment methods(RLHF, RRHF, DPO, IPO) and provide how to run examples.

## Common Notations

$x$ is a prompt, $y_w$ is better continuation and $y_l$ is worse continuation and $\pi$ (or $\rho$, $\mu$), $\pi_{ref}$ are behavior, reference model respectively. $\mathcal{D}$ is dataset distribution.

## RLHF Overview
As shown in [InstructGPT paper](https://arxiv.org/abs/2203.02155), we can train a reward model and reinforce a language model to follow human instructions better. We can first train reward model and SFT-LM with `hh-rlhf` dataset and `slimorca-dedup` dataset respectively. Then PPO-LM can be trained with trl library.

![workflow](assets/workflow.PNG)


### Train a Reward Model

we need pairwise comparison dataset to train a reward model. In the InstructGPT paper, authors used 4~9 ranked continuations on the same prompt. For instance, `A < B < C = D < E` is a ranked sequence and one can sample two arbitrary samples(A and C). Here, C wins over A on the human preference. Thus we model `logit of C - logit of A` to be the log odds of C being a better demonstration than A. `logit of X` can be computed by a linear head attached at the top of a transformer decoder. We use off-the-shelf dataset from [hh-rlhf](https://github.com/anthropics/hh-rlhf) by anthropic. This dataset is already flat so we don't need to worry about sampling schemes discussed in InstructGPT paper.

### Train a SFT(supervised finetuned) model (WIP)

We can train SFT model with standard next-token-prediction using [slimorca-dedup](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup). 

### Train a PPO model (WIP)

Now that we have a reward model and a SFT model, we can do reinforcement learning with off-the-shelf RL packages designed for language models. We use [trl](https://github.com/lvwerra/trl) to reinforce the SFT model. In PPO stage, we keep copy of SFT model for reference. This reference model allows behavior model to learn to increase human preference while avoiding reward hacking. Specifically, behavior model first generates a completion given prompt. Token distributions are kept close to reference model via minimising KL-divergence against reference model's token distribution. A completion is fed to reward model to get a reward score. KL term and reward score are summed and regarded as a reward for PPO algorithm.

Here's what we maximise: 

```math
J(\pi) = \mathbb{E}_\pi[r(x,y)] - \tau D_{KL}(\pi || \pi_{ref})
```

where

```math
D_{KL}(\pi || \pi_{ref}) = \mathbb{E}_{y \sim \pi(\cdot | x)} \left[\log \left( \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right) \right]
```

### Quickstart

- Prepare virtual environment (optional)

```
conda create -n easy-rlhf python=3.8
```

- Clone and install requirements

```
git clone https://github.com/DaehanKim/EasyRLHF.git
cd EasyRLHF
pip install .
```

- Unzip hh-rlhf dataset and train a reward model by using `rm_train` cmd

```
cd data
find . -name '*.gz' -print0 | xargs -0 gzip -d
rm_train --devices "0,1,2,3" \
--output_dir "outputs/my-model" \
--master_port 50000 \
--train_file data/helpful-base/train.jsonl,data/helpful-online/train.jsonl,data/helpful-rejection-sampled/train.jsonl \
--valid_file data/helpful-base/test.jsonl,data/helpful-online/test.jsonl,data/helpful-rejection-sampled/test.jsonl
```

- Alternatively, you can use `scripts/rm_train.sh` for more customized settings

### Notes
- Default model is gpt2-xl(1.5B) and the loss is binary cross entropy.
- Deepspeed config is in `configs/ds_config.yaml` where you can set your preferred distributed setting. Default is set to Zero-2 Parallelism.
- ToDo
  - [x] basic reward model training
  - [ ] basic SFT model training
  - [ ] basic PPO model training

## DPO Overview 

DPO avoids training a reward model and directly optimizes RLHF objective with respect to output vocab distribution. Here's what we optimize : 

```math
\min_\pi \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ -\log \sigma \left( \tau \log \left(\frac{\pi(y_w | x)}{\pi(y_l | x)} \right)  - \tau \log \left(\frac{\pi_{ref}(y_w | x)}{\pi_{ref}(y_l | x)} \right) \right) \right]
```

## IPO Overview

TBD

## RRHF Overview

RRHF stands for "rank responses from human feedback". The gist of this method is to align response probability towards human preference. More specifically, let's say we have k different responses $y_i$ of x sampled by policy $\rho_i (1 \le i \le k)$. $R(x, y_i) = r_i$ is each response's reward score(or rank). 

We can optimise following loss:

```math
\mathcal{L} = \sum_{r_i \lt r_j} \max(0, p_i - p_j) - \sum_t \log P_\pi (y_{i', t} | x , y_{i',\lt t})
```

where $y_{i'}$ indicates a response with highest score and

```math
p_i = \frac{\sum_t \log P_\pi(y_{i,t} | x, y_{i, \lt t})} {||y_i||}
```

is a length-normalized log probability.

Note that this method doesn't require scalar reward values but just a relative order of responses.

----

## References
- [InstructGPT paper](https://arxiv.org/abs/2203.02155)
- [trl](https://github.com/lvwerra/trl)
- [hh-rlhf](https://github.com/anthropics/hh-rlhf)
- [slimorca-dedup](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup)
- [RRHF](https://arxiv.org/abs/2304.05302)
- [DPO](https://arxiv.org/abs/2305.18290)
- [IPO](https://arxiv.org/abs/2310.12036#deepmind)

## License

This project just binds libraries and datasets from various sources, so is under license terms of corresponding sources. 
Binding script itself is licensed MIT.