#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import copy
from utils.sh_utils import SH2RGB
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    CA_score = 0
    CA_num = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, train=False, dropout_factor=args.dropout_factor)
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # if args.render_depth
        depth = 1.0 - (rendering['rendered_depth'] - rendering['rendered_depth'].min()) / (rendering['rendered_depth'].max() - rendering['rendered_depth'].min())
        depth_est = (1 - depth * rendering["rendered_alpha"]).squeeze().cpu().numpy()
        depth_map = visualize_cmap(depth_est, np.ones_like(depth_est), matplotlib.colormaps['turbo'], curve_fn=depth_curve_fn).copy()
        np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['rendered_depth'][0].detach().cpu().numpy())
        depth_map = torch.as_tensor(depth_map).permute(2,0,1)
        torchvision.utils.save_image(depth_map, os.path.join(render_path, view.image_name + '_depth.png'))
        
        # render GS pointmap
        gaussians1 = copy.deepcopy(gaussians)
        rendering1 = render(view, gaussians1, pipeline, background, render_pts=True, render_pts_scale=0.01, train=False, dropout_factor=args.dropout_factor)
        torchvision.utils.save_image(rendering1["render"], os.path.join(render_path, '{0:05d}'.format(idx) + "_points.png"))
        
        # renders difference between different GS subsets and CA variance map
        rendered_images = []
        visi_areas = []
        for i in range(5):
            rendering_drop = render(view, gaussians, pipeline, background, dropout_factor=1-(1-args.dropout_factor)/2, train=True) # 0.5 if not dropout during training
            render_image = rendering_drop["render"]
            visi_area = rendering_drop["visi_area"]
            rendered_images.append(render_image.cpu())
            visi_areas.append(visi_area)
            # print(visi_area.sum())
            # torchvision.utils.save_image(render_image, os.path.join(render_path, f'{idx:05d}_dropout_{i}.png'))
            
        def compute_common_visible_mask(visi_areas):
            common_mask = visi_areas[0].clone()
            for mask in visi_areas[1:]:
                common_mask = common_mask & mask
            return common_mask.cpu()
        
        visi_area_common = compute_common_visible_mask(visi_areas).unsqueeze(0)
        rendered_images_tensor = torch.stack(rendered_images, dim=0)
        rendered_images_tensor *= visi_area_common
        images_np = rendered_images_tensor.numpy()
        images_np_hw3 = np.transpose(images_np, (0, 2, 3, 1))
        pixelwise_var = np.var(images_np_hw3, axis=0)  # shape: (H, W, 3)
        pixelwise_var_mean = np.mean(pixelwise_var, axis=2)  # RGB var grayscale，shape: (H, W)
        common_mask = visi_area_common.squeeze(0).squeeze(0)
        pixelwise_var_mean_mean = pixelwise_var_mean[common_mask].mean().item()
        # torchvision.utils.save_image(common_mask.unsqueeze(0).float(), os.path.join(render_path, f'{idx:05d}_commonmask.png'))
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        for i in range(5):
            img = np.transpose(images_np[i], (1, 2, 0))  # (H, W, 3)
            img = np.clip(img, 0, 1)
            axs[i // 3, i % 3].imshow(img)
            axs[i // 3, i % 3].set_title(f"Dropout Rendering {i+1}")
            axs[i // 3, i % 3].axis('off')

        im = axs[1, 2].imshow(pixelwise_var_mean, cmap='inferno')
        axs[1, 2].set_title(f"Pixelwise Variance Heatmap (CA: {pixelwise_var_mean_mean:.6f})")
        axs[1, 2].axis('off')
        plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(render_path, '{0:05d}'.format(idx) + f'_dropout_comparison_variance.png'))
        plt.close()
        
        CA_num += 1
        CA_score += pixelwise_var_mean_mean
        
    CA_score /= CA_num
    
    print(f"{name} CA_score:{CA_score}")
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        print(f"GS num: {gaussians._xyz.shape}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_views", default=3, type=int)
    parser.add_argument("--dropout_factor", type=float, default=0.2) # 0.2
    parser.add_argument("--dataset_name", default="LLFF", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)