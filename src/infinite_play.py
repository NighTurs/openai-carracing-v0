import os
import argparse
from src.config import config
from src.train import RUNS_DIR
from src.env_wrap import make_env
from stable_baselines3 import PPO


def infinite_play(run_name: str):
    run_dir = os.path.join(RUNS_DIR, run_name)
    cfg = config['preprocess']
    env = make_env(seed=0,
                   n_envs=1,
                   run_dir=run_dir,
                   frame_skip=cfg['frame_skip'],
                   frame_stack=cfg['frame_stack'],
                   is_eval=True)
    model = PPO.load(os.path.join(RUNS_DIR, run_name, 'best_model.zip'))
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of the training run')
    args = parser.parse_args()
    infinite_play(args.run_name)
