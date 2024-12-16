#!/usr/bin/env bash
set -Eeuo pipefail
env_name="robot_learning_torchrl"

eval "$(conda "shell.$(basename "${SHELL}")" hook)"
conda create -n ${env_name} python=3.11 -y
conda activate ${env_name}
ROOT_FOLDER="$(dirname "$0")"
cd "${ROOT_FOLDER}"

cd ~/Projects/franka_sim
pip install -e .

pip install tensordict==0.6.2
pip install torchrl==0.6.0
pip install tqdm hydra-core wandb moviepy
