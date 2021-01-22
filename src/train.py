import argparse
import os
from typing import Dict, Any
from src.config import dump as dump_config
from src.config import config as default_config

from stable_baselines3 import PPO
from src.env_wrap import make_env
from src.common import get_run_dir


def train(run_name: str, config: Dict[str, Any]):
    cfg_t = config['train']
    cfg_p = config['preprocess']
    run_dir = get_run_dir(run_name)
    dump_config(run_dir, config)
    os.makedirs(run_dir, exist_ok=False)

    def _make_env(n_envs: int, is_eval: bool):
        return make_env(seed=0,
                        n_envs=n_envs,
                        run_dir=run_dir,
                        frame_skip=cfg_p['frame_skip'],
                        frame_stack=cfg_p['frame_stack'],
                        is_eval=is_eval)

    train_env = _make_env(cfg_t['n_envs'], False)
    eval_env = _make_env(1, False)
    model = PPO('CnnPolicy',
                train_env,
                n_steps=cfg_t['n_steps'],
                n_epochs=cfg_t['n_epochs'],
                batch_size=cfg_t['batch_size'],
                learning_rate=cfg_t['lr'],
                tensorboard_log=os.path.join(run_dir, 'tb'))
    model.learn(cfg_t['total_steps'],
                eval_env=eval_env,
                eval_freq=cfg_t['eval_freq'] // cfg_t['n_envs'],
                n_eval_episodes=cfg_t['n_eval_eps'],
                eval_log_path=run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of the training run')
    args = parser.parse_args()
    train(args.run_name, default_config)
