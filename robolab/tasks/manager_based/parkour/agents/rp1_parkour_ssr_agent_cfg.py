from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.amp.agents.rpo_amp_agent_cfg import (
    RslRlAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from robolab.tasks.manager_based.parkour.mdp.foothold_prediction import (
    FootholdGridCfg,
    FootholdPredictorCfg,
)
from robolab.tasks.manager_based.parkour.mdp.symmetry import rp1


@configclass
class RslRlPpoEncoderMoEActorMultiCriticCfg:
    class_name: str = "EncoderMoEActorMultiCritic"
    init_noise_std: float = 1.0
    num_moe_experts: int = 5
    # SSR Table 8: Expert MLP [1024, 512, 128], Gate MLP hidden size 128.
    moe_gate_hidden_dims: list[int] = [128]
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [512, 256, 128]
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = None
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
    encoder_onnx_sequential_idx: int | None = None
    actor_onnx_filename: str = "policy_parkour.onnx"


@configclass
class RslRlMultiRewardPpoAmpAlgorithmCfg(RslRlPpoAmpAlgorithmCfg):
    class_name: str = "MultiRewardPPOAMP"
    num_reward_heads: int = 3
    advantage_weights: list[float] = [1.0, 0.25, 0.2]


@configclass
class RP1ParkourSSRAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rp1_parkour_ssr"
    wandb_project = "rp1_parkour_ssr"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoEncoderMoEActorMultiCriticCfg()
    foothold_imagination: FootholdPredictorCfg = FootholdPredictorCfg(
        enabled=True,
        hidden_dims=[256, 128],
        learning_rate=5.0e-4,
        weight_decay=1.0e-5,
        ema_decay=0.99,
        grid=FootholdGridCfg(
            sigma_min=0.01,
            sigma_max=0.25,
            expectation_grid_size=5,
            expectation_std_range=2.0,
            expectation_eval_chunk_size=64,
        ),
        nll_loss_coef=1.0,
        max_pending_steps=48,
        pending_sample_stride=1,
        train_pending_tail_steps=12,
        train_buffer_capacity=100_000,
        batch_size=4096,
        updates_per_iteration=8,
        min_train_samples=25_000,
        curriculum_level_threshold=1.0,
        enable_xy_rmse_threshold=0.05,
        clear_train_buffer_on_resume=True,
    )
    algorithm = RslRlMultiRewardPpoAmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rp1.compute_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=5.0,
            disc_trunk_weight_decay=1.0e-5,
            disc_linear_weight_decay=1.0e-3,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="ReLU",
                style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            loss_type="LSGAN",
        ),
    )
