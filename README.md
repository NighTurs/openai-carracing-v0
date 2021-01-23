# OpenAI CarRacing-v0 Agent

Reinforcement learning agent for 
[OpenAI gym CarRacing-v0 task](https://gym.openai.com/envs/CarRacing-v0/).

## Setup

Prerequisites:
* python 3.9
* pipenv
* swig

Setup virtual environment:
```
pipenv install --ignore-pipfile
```

Activate virtual environment:
```
pipenv shell
```

## Training

Check configuration in [config.yaml](config.yaml).

> Set `n_envs` to number of CPU cores on machine. And adjust `n_steps` 
> so that `n_envs` * `n_steps` roughly stays the same 

To start training:
```
python -m src.train --run_name some_name
```

To monitor training:
```
tensorboard --logdir ./runs/some_name/tb
```

## Demo

Infinite play using trained agent:
```
python -m src.demo.infinite_play --run_name some_name
```

Record a video using trained agent (will be in `./runs/some_name/video`):
```
python -m src.demo.record_video --run_name some_name
```