"""
Generate a large batch of channel samples from a model and save them as a large
numpy array. This can be used to produce samples for evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # 处理图像尺寸
    if args.image_height is not None and args.image_width is not None:
        image_size = (int(args.image_height), int(args.image_width))
    else:
        image_size = int(args.image_size) if args.image_size is not None else None

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # 根据图像尺寸设置采样形状
        if isinstance(image_size, tuple):
            height, width = image_size
            sample_shape = (int(args.batch_size), int(args.in_channels), int(height), int(width))
        else:
            size = int(image_size) if image_size is not None else int(args.image_size)
            sample_shape = (int(args.batch_size), int(args.in_channels), size, size)
            
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        in_channels=2,  # 默认输入通道数
        image_size=64,  # 默认图像大小
        image_height=None,  # 新增：图像高度
        image_width=None,  # 新增：图像宽度
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
