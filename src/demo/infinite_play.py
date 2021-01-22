import argparse
from src.demo.common import load_model


def infinite_play(run_name: str):
    env, model = load_model(run_name)
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
