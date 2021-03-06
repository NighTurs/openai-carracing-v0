import os
from typing import Tuple
from src.common import get_run_dir
from src.config import load
from src.env_wrap import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def load_model(run_name: str, model_file: str) -> Tuple[VecEnv, PPO]:
    run_dir = get_run_dir(run_name)
    cfg = load(run_dir)['preprocess']
    env = make_env(seed=123,
                   n_envs=1,
                   run_dir=run_dir,
                   frame_skip=cfg['frame_skip'],
                   frame_stack=cfg['frame_stack'],
                   is_eval=True)
    model = PPO.load(os.path.join(run_dir, model_file))
    return env, model
