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
    num_moe_experts: int = 1
    moe_gate_hidden_dims: list[int] = []
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    activation: str = "elu"
    actor_encoder_obs_groups: list[str] = ["depth_image"]
    critic_encoder_obs_groups: list[str] = None
    # Depth 8x18x32, memory-friendly: downsample early, avoid wide maps at full res.
    # Spatial ≈ 18x32 → 9x16 → 4x8 → 4x8 before MLP → 256.
    encoder_cfg: dict = {
        "channels": [16, 32, 64],
        "kernel_sizes": [3, 3, 3],
        "strides": [2, 2, 1],
        "hidden_sizes": [256, 256],
        "output_size": 256,
        "paddings": [1, 1, 1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
    }
    encoder_onnx_stems: dict[str, str] = {"depth_image": "depth_encoder"}
    encoder_onnx_sequential_idx: int | None = None
    actor_onnx_filename: str = "policy_ssr.onnx"
    # Actor features: [flat_proprio_history, v_hat, f_hat, h_hat, depth_latent].
    enable_ssr_estimation: bool = True
    proprio_history_length: int = 8
    estimation_proprio_encoder_hidden_dims: list[int] = [512, 256]
    estimation_proprio_latent_dim: int = 256
    estimation_estimator_hidden_dims: list[int] = [512, 256] # velocity_estimator、foothold_estimator
    estimation_decoder_hidden_dims: list[int] = [256, 128] # foot_height_decoder
    estimation_actor_use_proprio_latent: bool = False
    enable_actor_foothold: bool = True
    enable_actor_foot_height: bool = True
    estimation_foot_height_actor_dim: int = 16
    # Wait for privileged predictor reward_enabled before SSR foothold aux / f_hat.
    gate_foothold_estimation_on_predictor_ready: bool = True
    aux_velocity_coef: float = 1.0
    aux_foot_height_coef: float = 1.0
    aux_foothold_coef: float = 2.0


@configclass
class RslRlMultiRewardPpoAmpAlgorithmCfg(RslRlPpoAmpAlgorithmCfg):
    class_name: str = "MultiRewardPPOAMP"
    num_reward_heads: int = 3
    advantage_weights: list[float] = [1.0, 0.8, 0.2]
    reward_head_names: list[str] = ["locomotion", "foothold", "style"]
    enable_aux_loss: bool = True
    aux_loss_coef: float = 1.0


@configclass
class RP1SSRAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rp1_ssr"
    wandb_project = "rp1_ssr"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoEncoderMoEActorMultiCriticCfg()
    foothold_imagination: FootholdPredictorCfg = FootholdPredictorCfg(
        enabled=True,
        hidden_dims=[512, 256, 128],
        learning_rate=5.0e-4,
        weight_decay=1.0e-5,
        ema_decay=0.99,
        grid=FootholdGridCfg(
            sigma_min=0.01,
            sigma_max=0.25,
            yaw_sigma_min=0.05,
            yaw_sigma_max=0.5,
            expectation_grid_size=5,
            expectation_std_range=2.0,
            expectation_eval_chunk_size=64,
        ),
        nll_loss_coef=1.0,
        max_pending_steps=48,
        pending_sample_stride=1,
        train_pending_tail_steps=24,
        train_buffer_capacity=100_000,
        batch_size=4096,
        updates_per_iteration=8,
        min_train_samples=25_000,
        curriculum_level_threshold=1.0,
        enable_xy_rmse_threshold=0.05,
        enable_yaw_rmse_threshold=0.15,
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
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
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
