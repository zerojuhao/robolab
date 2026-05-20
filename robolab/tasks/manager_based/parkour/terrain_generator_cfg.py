# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import robolab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "perlin_rough": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "perlin_rough_stand": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "square_gaps": terrain_gen.PerlinSquareGapTerrainCfg(
            proportion=0.1,
            gap_distance_range=(0.1, 0.40),
            gap_depth=(0.4, 0.6),
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_32": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_30": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_28": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_inv_32": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_inv_30": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_inv_28": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        "pyramid_stairs_inv_high_ground_aligned": terrain_gen.PerlinInvertedPyramidStairsGroundAlignedTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.0,
            border_width=2.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-4.0, 4.0),
                ),
            },
        ),
        # "stepping_stones": terrain_gen.PerlinSteppingStonesTerrainCfg(
        #     proportion=0.1,
        #     stone_height_max=0.0,
        #     stone_width_range=(0.3, 0.4),
        #     stone_distance_range=(0.1, 0.2),
        #     platform_width=1.5,
        #     border_width=1.0,
        #     wall_prob=[0.3, 0.3, 0.3, 0.3],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
        #         noise_scale=0.05,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],
        #             max_height_diff=0.05,
        #             x_range=(3.7, 3.7),
        #             y_range=(-4.0, 4.0),
        #         ),
        #     },
        # ),
        # "boxes": terrain_gen.PerlinDiscreteObstaclesTerrainCfg(
        #     proportion=0.10,
        #     num_obstacles=20,
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(0.8, 1.5),
        #     obstacle_height_range=(0.05, 0.2),
        #     platform_width=1.5,
        #     border_width=0.0,
        #     wall_prob=[0.3, 0.3, 0.3, 0.3],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
        #         noise_scale=0.05,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),
        # "mesh_boxes": terrain_gen.PerlinMeshRandomMultiBoxTerrainCfg(
        #     proportion=0.10,
        #     box_height_mean=[0.05, MAX_STAIR_HEIGHT],
        #     box_height_range=0.05,
        #     box_length_mean=0.4,
        #     box_length_range=0.1,
        #     box_width_mean=0.4,
        #     box_width_range=0.1,
        #     platform_width=1.5,
        #     generation_ratio=0.3,
        #     no_perlin_at_obstacle=True,
        #     wall_prob=[0.3, 0.3, 0.3, 0.3],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05),
        #     },
        # ),
        "hf_pyramid_slope_inv": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.2),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
    },
)

# ROUGH_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
#     seed=0,
#     size=(8.0, 8.0),
#     border_width=3,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.05,
#     vertical_scale=0.005,
#     slope_threshold=1.0,
#     use_cache=False,
#     curriculum=True,
#     sub_terrains={
#         "inv_pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.05,
#             step_height_range=(0.05, 0.2),
#             step_width=0.25,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "inv_pyramid_stairs_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.05,
#             step_height_range=(0.05, 0.2),
#             step_width=0.35,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_25": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.05,
#             step_height_range=(0.05, 0.2),
#             step_width=0.25,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_35": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.05,
#             step_height_range=(0.05, 0.2),
#             step_width=0.35,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.05, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0
#         ),
#         "inv_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.05, slope_range=(0.1, 0.3), border_width=1.0, platform_width=2.0, inverted=True
#         ),
#         "grid": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.1, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=1.0
#         ),
#         "high_platform": terrain_gen.MeshPitTerrainCfg(
#             proportion=0.1, pit_depth_range=(0.1, 0.3), platform_width=2.0, double_pit=True
#         ),
#         "star": terrain_gen.MeshStarTerrainCfg(
#             proportion=0.1, num_bars=12, bar_width_range=(0.25, 0.4), bar_height_range=(10.0, 10.0), platform_width=2.0
#         ),
#         "gap": terrain_gen.MeshGapTerrainCfg(
#             proportion=0.15, gap_width_range=(0.1, 0.3), platform_width=2.0
#         ), # points in gap are nan
#         "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
#             proportion=0.15,
#             stone_height_max=0.0,
#             stone_width_range=(0.3, 0.4),
#             stone_distance_range=(0.1, 0.2),
#             platform_width=2.0,
#             border_width=1.0,
#         ),
#     },
# )
