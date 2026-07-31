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
class RslRlPpoEncoderMoEActorCriticCfg:
    class_name: str = "EncoderMoEActorCritic"
    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = []
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    actor_obs_normalization: bool = False  # NOTE!: DO NOT SET TO TRUE, OR IT WILL CAUSE THE ROBOT TO CRASH IN REAL-WORLD DEPLOYMENT CAUSE WE USE THE LATENT ENCODER FOR DEPTH OBSERVATIONS!
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
    encoder_onnx_sequential_idx: int = 0


@configclass
class RP1ParkourAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rp1_parkour"
    wandb_project = "rp1_parkour"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoEncoderMoEActorCriticCfg()
    foothold_imagination: FootholdPredictorCfg = FootholdPredictorCfg(
        enabled=True,
        hidden_dims=[256, 128],
        learning_rate=5.0e-4,
        weight_decay=1.0e-5,
        ema_decay=0.99,
        grid=FootholdGridCfg(
            # Bbox spans must be integer multiples of resolution (both axes 1.2 m at center_y=0.15).
            resolution=0.03,
            reach_center=(0.0, 0.15),
            reach_radii=(0.6, 0.45),
            reward_quality_top_k=64,
            reward_quality_eval_chunk_size=64,
            reward_unselected_mass_penalty=1.0,
            residual_span_cells=1.0,
        ),
        # Prefer classification over residual: mode XY is limited by top1, not teacher.
        classification_loss_coef=1.0,
        residual_loss_coef=0.25,
        residual_neighbor_weight=0.1,
        ce_neighbor_weight=0.25,
        classify_in_reach_only=True,
        max_pending_steps=48,
        pending_sample_stride=1,
        train_pending_tail_steps=12,
        train_buffer_capacity=100_000,
        batch_size=4096,
        updates_per_iteration=8,
        min_train_samples=25_000,
        curriculum_level_threshold=3.0,
        enable_xy_rmse_threshold=0.04,
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
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
