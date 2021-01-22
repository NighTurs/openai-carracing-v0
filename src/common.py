import os

RUNS_DIR = './runs'


def get_run_dir(run_name: str) -> str:
    return os.path.join(RUNS_DIR, run_name)
