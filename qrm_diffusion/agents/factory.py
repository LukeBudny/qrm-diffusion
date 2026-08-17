from __future__ import annotations

import torch

from .controller import RecedingHorizonStepPolicy


def create_policy(agent_config, device=None) -> RecedingHorizonStepPolicy:
    policy = RecedingHorizonStepPolicy(agent_config.policy.hidden_dim)
    with torch.no_grad():
        policy.output.bias.fill_(agent_config.initialization.schedule_action)
    return policy if device is None else policy.to(device)
