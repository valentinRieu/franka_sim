#!/usr/bin/env bash
set -Eeuo pipefail
env_name="robot_learning_torchrl"

eval "$(conda "shell.$(basename "${SHELL}")" hook)"
conda create -n ${env_name} python=3.10 -y
conda activate ${env_name}
ROOT_FOLDER="$(dirname "$0")"
cd "${ROOT_FOLDER}"

cd ~/Projects/franka_sim
pip install -e .

# pip install 'skrl[torch]@git+https://github.com/Toni-SM/skrl.git@57f60df'
pip install tensordict-nightly
pip install torchrl-nightly
pip install tqdm hydra-core wandb moviepy
