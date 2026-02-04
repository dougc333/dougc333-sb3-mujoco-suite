# rl_core/policies.py
from __future__ import annotations

import torch as th
import torch.nn as nn
from torch.nn.utils import spectral_norm
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    # Standard RL trick
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def maybe_spectral(layer: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(layer) if enabled else layer


class SpectralActorCriticPolicy(ActorCriticPolicy):
    """
    SB3 drop-in PPO policy:
      - spectral norm on policy/value MLP layers (optional)
      - orthogonal init (optional)
    """

    def __init__(
        self,
        *args,
        spectral: bool = True,
        orthogonal: bool = True,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        **kwargs,
    ):
        self._spectral = spectral
        self._orthogonal = orthogonal
        self._net_arch = net_arch or [256, 256]
        self._activation_fn = activation_fn
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Build policy/value nets manually so we can wrap Linear with spectral_norm
        def make_mlp():
            layers: list[nn.Module] = []
            last = self.features_dim
            for h in self._net_arch:
                lin = nn.Linear(last, h)
                lin = maybe_spectral(lin, self._spectral)
                layers += [lin, self._activation_fn()]
                last = h
            return nn.Sequential(*layers)

        policy_net = make_mlp()
        value_net = make_mlp()

        # SB3 expects an MlpExtractor-like object providing latents
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[],  # unused
            activation_fn=self._activation_fn,
            device=self.device,
        )
        # override internal nets
        self.mlp_extractor.policy_net = policy_net
        self.mlp_extractor.value_net = value_net

        if self._orthogonal:
            # Typical gains: sqrt(2) for hidden, 0.01 for action head handled by SB3
            self.mlp_extractor.apply(lambda m: orthogonal_init(m, gain=th.sqrt(th.tensor(2.0)).item()))
