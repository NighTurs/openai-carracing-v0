from typing import Callable

import gym
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, \
    SubprocVecEnv, VecFrameStack

ENV_NAME = 'CarRacing-v0'


def make_env(seed: int,
             n_envs: int,
             run_dir: str,
             frame_skip: int,
             frame_stack: int,
             is_eval: bool = False) -> VecEnv:
    """
    Makes vectorized env with required wrappers
    :param seed: Random seed
    :param n_envs: Number of environment to run in parallel
    :param run_dir: Run directory
    :param frame_skip: Skip every nth frame
    :param frame_stack: Stack n frames together
    :param is_eval: True if used for evaluation
    :return: Vectorized env
    """
    if n_envs == 1:
        env = DummyVecEnv([_env_fn(seed,
                                   run_dir,
                                   frame_skip,
                                   is_eval)])
    else:
        env = SubprocVecEnv([_env_fn(seed + i,
                                     run_dir,
                                     frame_skip,
                                     is_eval) for i in range(n_envs)])
    if frame_stack > 0:
        return VecFrameStack(env, n_stack=4)
    else:
        return env


def _env_fn(seed: int,
            run_dir: str,
            frame_skip: int,
            is_eval: bool = False) -> Callable[[], gym.Env]:
    def _inner() -> gym.Env:
        env = gym.make(ENV_NAME, verbose=0)
        env.seed(seed)
        if not is_eval:
            env = Monitor(env, run_dir)
        env = GrayScaleObservation(env, keep_dim=True)
        if frame_skip > 0:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        return env

    return _inner
