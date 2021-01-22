import os
from typing import Dict, Any

import yaml

CONFIG_FILE = 'config.yaml'


def load(dir_path: str) -> Dict[str, Any]:
    with open(os.path.join(dir_path, CONFIG_FILE), 'r') as f:
        return yaml.safe_load(f)


def dump(dir_path: str, config: Dict[str, Any]):
    with open(os.path.join(dir_path, CONFIG_FILE), 'w') as f:
        yaml.safe_dump(config, f)


config = load('./')
