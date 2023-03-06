from setuptools import setup, find_packages

setup(
    name="easy-rlhf",
    version="0.1.0",
    description="This repo aims to providing step-by-step instructions and resources to train your own RLHF models, using only off-the-shelf solutions and packages (i.e. HF Trainer, HF Datasets, Deepspeed, trl and whatnot).",
    author="Daehan Kim",
    author_email="kdh5852@gmail.com",
    url="https://github.com/DaehanKim/EasyRLHF",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "datasets",
        "transformers",
        "torch",
        "deepspeed",
        "accelerate",
        "torch",
        "wandb",
    ],
    entry_points={
        "console_scripts": [
            "rm_train=EasyRLHF.cmd_main:rm_train",
        ],
    },
    package_data={
        "configs": ["configs/ds_config.json"],
    },
)
