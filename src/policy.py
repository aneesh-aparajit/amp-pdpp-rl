import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# impo


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy(ActorCriticPolicy):
    def __init__(self, hidden_size, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)
        self.features_extractor = PolicyNetwork(
            obs_size=self.observation_space.shape[0],
            hidden_size=hidden_size,
            action_size=self.features_dim
        )
