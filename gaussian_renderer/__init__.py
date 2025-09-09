#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, override_color = None, dropout_factor=0.0, 
           sigma_noise = 0.0, render_pts=False, render_pts_scale=0.1, train=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, 
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # init random dropout mask
    if dropout_factor > 0.0 and train:
        dropout_mask = torch.rand(pc.get_opacity.shape[0], device=pc.get_opacity.device).cuda()
        dropout_mask = dropout_mask < (1 - dropout_factor)
    else:
        dropout_mask = torch.rand(pc.get_opacity.shape[0], device=pc.get_opacity.device).cuda()
        dropout_mask = dropout_mask < 1.1

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    if render_pts:
        scales = torch.full_like(scales, render_pts_scale).cuda()
        
    # 1. randomly dropout 3DGS points during training
    if dropout_factor > 0.0 and train:
        means3D      = means3D[dropout_mask]
        means2D      = means2D[dropout_mask]
        shs          = shs[dropout_mask]
        opacity      = opacity[dropout_mask]
        scales       = scales[dropout_mask]
        rotations    = rotations[dropout_mask]
    elif (not train):
        # scale oapcity for test stage rendering
        opacity *= 1 - dropout_factor
        
    # 2. add noise to opacity during training
    if train and sigma_noise > 0.0:
        # sigma_noise = 0.8  # 0.8
        epsilon_opacity = torch.randn_like(opacity, device=opacity.device) * sigma_noise
        epsilon_opacity = torch.clamp(epsilon_opacity, min=-sigma_noise, max=sigma_noise)  # 根据实际训练经验设定合理范围
        opacity = torch.clamp(opacity * (1.0 + epsilon_opacity), min=0.0, max=1.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rendered_depth": depth,
            "rendered_alpha": alpha,
            "visi_area": alpha > 0.8,
            "dropout_mask": dropout_mask
            }
