import os
import argparse
from src.common import get_run_dir
from src.demo.common import load_model
from stable_baselines3.common.vec_env import VecVideoRecorder


def record_video(run_name: str, video_length: int):
    run_dir = get_run_dir(run_name)
    env, model = load_model(run_name)
    env = VecVideoRecorder(env, os.path.join(run_dir, 'video'),
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix='video')
    obs = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        env.render()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',
                        required=True,
                        help='Name of the training run')
    parser.add_argument('--video_length',
                        type=int,
                        default=1000,
                        help='Timesteps to record')
    args = parser.parse_args()
    record_video(args.run_name, args.video_length)
