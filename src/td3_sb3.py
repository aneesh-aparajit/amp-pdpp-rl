import os

import gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor

from config import CFG
from dataset import get_dataset
from env import ParkinsonEnv

models_dir = f'../artifacts/best_models/TD3/'
logdir = f'../../artifacts/logs/TD3/'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

X_train, X_test, y_train, y_test = get_dataset(exp_name='TD3_sb3')
train_env = ParkinsonEnv(dataset=(X_train, y_train),
                         samples_per_episode=CFG.sample_per_episode)
valid_env = ParkinsonEnv(dataset=(X_test, y_test),
                         samples_per_episode=CFG.sample_per_episode)

train_env = DummyVecEnv([lambda: Monitor(
    train_env, 
    f'../artifacts/logs/TD3/'
)])
valid_env = DummyVecEnv([lambda: Monitor(
    valid_env, 
    f'../artifacts/logs/TD3/'
)])

model = TD3(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log="../artifacts/logs/TD3"
)
callback = EvalCallback(eval_env=valid_env, eval_freq=1000, best_model_save_path=models_dir, verbose=1)


for i in range(30):
    model.learn(
        total_timesteps=CFG.TIMESTAMPS, 
        reset_num_timesteps=False, 
        tb_log_name="../artifacts/logs/TD3/",
        callback=callback
    )

model.save(os.path.join(models_dir, 'final_model'))
