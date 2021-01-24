import argparse
from src.demo.common import load_model


def infinite_play(run_name: str, model_file: str):
    env, model = load_model(run_name, model_file)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of the training run')
    parser.add_argument('--model_file', default='best_model.zip',
                        help='Model file name')
    args = parser.parse_args()
    infinite_play(args.run_name, args.model_file)
