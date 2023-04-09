import os

import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CFG
from dataset import get_dataset
from env import ParkinsonEnv
from policy import Policy

models_dir = f'../artifacts/best_models/A2C_Custom/'
logdir = f'../../artifacts/logs/A2C_Custom/'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

X_train, X_test, y_train, y_test = get_dataset(exp_name='A2C_custom')
train_env = ParkinsonEnv(dataset=(X_train, y_train),
                         samples_per_episode=CFG.sample_per_episode)
valid_env = ParkinsonEnv(dataset=(X_test, y_test),
                         samples_per_episode=CFG.sample_per_episode)

train_env = DummyVecEnv([lambda: Monitor(
    train_env,
    f'../artifacts/logs/A2C_Custom/'
)])
valid_env = DummyVecEnv([lambda: Monitor(
    valid_env,
    f'../artifacts/logs/A2C_Custom/'
)])

policy = Policy(
    observation_space=train_env.observation_space,
    hidden_size=CFG.hidden_size,
    action_space=train_env.action_space,
    lr_schedule=lambda lr: lr,
)

# optimizer = torch.optim.Adam(policy.parameters(), lr=CFG.learning_rate)

model = A2C(
    policy=policy,
    # optimizer=optimizer,
    env=train_env,
    verbose=1,
    tensorboard_log="../artifacts/logs/A2C_Custom"
)
callback = EvalCallback(eval_env=valid_env, eval_freq=1000,
                        best_model_save_path=models_dir, verbose=1)


for i in range(30):
    model.learn(
        total_timesteps=CFG.TIMESTAMPS,
        reset_num_timesteps=False,
        tb_log_name="../artifacts/logs/A2C_Custom/",
        callback=callback
    )

model.save(os.path.join(models_dir, 'final_model'))
