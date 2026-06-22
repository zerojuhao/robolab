# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoEncoderActorCriticCfg:
    class_name: str = "EncoderActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = ["depth_image"]
    encoder_cfg: dict = {
        "channels": [4],
        "kernel_sizes": [3],
        "strides": [1],
        "hidden_sizes": [256, 256],
        "output_size": 128,
        "paddings": [1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
    }
    encoder_onnx_stems: dict[str, str] = {"depth_image": "depth_encoder"}
    encoder_onnx_sequential_idx: int = 0


@configclass
class RPOPerceptiveRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "rpo_perceptive"
    wandb_project = "rpo_perceptive"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoEncoderActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
