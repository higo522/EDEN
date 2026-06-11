from src.models import load_model
from src.datasets import load_dataset
from src.utils import CalMetrics, InputPadder
from src.transport import create_transport, Sampler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
import diffusers
import torch
import argparse
from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import logging
import os
from glob import glob
import yaml
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def build_visualization_name(dataset, sample_index):
    sample_meta = dataset.meta_data[sample_index][0]
    if isinstance(sample_meta, Path):
        data_root = getattr(dataset, "data_root", None)
        if isinstance(data_root, Path):
            try:
                relative_path = sample_meta.relative_to(data_root)
            except ValueError:
                relative_path = sample_meta
        else:
            relative_path = sample_meta
        return "__".join(relative_path.with_suffix("").parts) + ".png"
    return f"sample_{sample_index:07d}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_eden.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        update_args = yaml.unsafe_load(f)
    parser.set_defaults(**update_args)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU!"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model_name = args.model_name
    output_dir = f"{args.output_dir}/eval-{model_name}"
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        experiment_index = len(glob(f"{output_dir}/*"))
        experiment_dir = f"{output_dir}/{experiment_index:03d}"
        visualization_dir = f"{experiment_dir}/visualization_results"
        os.makedirs(visualization_dir, exist_ok=True)
        if args.save_generated_frames:
            blended_dir = f"{visualization_dir}/blended_input"
            gt_dir = f"{visualization_dir}/ground_truth"
            generated_dir = f"{visualization_dir}/generated"
            os.makedirs(blended_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(generated_dir, exist_ok=True)
        evaluation_dir = f"{experiment_dir}/evaluation_results"
        os.makedirs(evaluation_dir, exist_ok=True)
        logging.basicConfig(
            format="[\033[34m%(asctime)s\033[0m] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info(f"Experiment directory created at {experiment_dir}")

    if args.global_seed is not None:
        set_seed(args.global_seed)

    # load dataset
    local_batch_size = args.dataloader["batch_size"]
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, **args.dataset_args[dataset_name])
    dataloader = DataLoader(dataset, **args.dataloader)
    dataset_len = len(dataset)
    steps_one_epoch = dataset_len // (local_batch_size * accelerator.num_processes)
    logger.info(f"Dataset {dataset_name} contains {dataset_len:,} triplets")

    # load model
    model = load_model(model_name, **args.model_args)
    logger.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")
    model.load_state_dict(ckpt["eden"])
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
    cal_metrics = CalMetrics()

    model, dataloader = accelerator.prepare(model, dataloader)
    raw_model = accelerator.unwrap_model(model)

    # begin training
    model.eval()
    steps = 0
    saved_triplet_index = 0
    results = {"PSNR": 0., "SSIM": 0., "LPIPS": 0., "FloLPIPS": 0., "L1": 0.}
    logger.info(f"Evaluating for {steps_one_epoch} steps...")
    for _, batch in enumerate(dataloader):
        frames = batch / 255.
        frame_0, frame_1, gt = frames[:, 0, ...], frames[:, 1, ...], frames[:, 2, ...]
        difference = ((torch.mean(torch.cosine_similarity(frame_0, frame_1), dim=[1, 2]) - args.cos_sim_mean) / args.cos_sim_std).unsqueeze(1).to(accelerator.device)
        img_size = [frame_0.shape[2], frame_0.shape[3]]
        padder = InputPadder(img_size)
        cond_frames = padder.pad(torch.cat((frame_0, frame_1), dim=0))
        with torch.no_grad():
            b, _, h, w = cond_frames.shape
            noise = torch.randn([b // 2, h // 32 * w // 32, args.model_args["latent_dim"]]).to(accelerator.device)
            denoise_kwargs = {"cond_frames": cond_frames, "difference": difference}
            samples = sample_fn(noise, raw_model.denoise, **denoise_kwargs)[-1]
            denoise_latents = samples / args.vae_scaler + args.vae_shift
            generated_frames = raw_model.decode(denoise_latents)
            generated_frames = padder.unpad(generated_frames.clamp(0., 1.))
        psnr = cal_metrics.cal_psnr(generated_frames, gt)
        ssim = cal_metrics.cal_ssim(generated_frames, gt)
        lpips = cal_metrics.cal_lpips(generated_frames, gt)
        flolpips = cal_metrics.cal_flolpips(generated_frames, gt, frame_0, frame_1)
        l1 = torch.mean(torch.abs(generated_frames - gt))
        results["PSNR"] += accelerator.gather(psnr.repeat(local_batch_size)).mean().item()
        results["SSIM"] += accelerator.gather(ssim.repeat(local_batch_size)).mean().item()
        results["LPIPS"] += accelerator.gather(lpips.repeat(local_batch_size)).mean().item()
        results["FloLPIPS"] += accelerator.gather(flolpips.repeat(local_batch_size)).mean().item()
        results["L1"] += accelerator.gather(l1.repeat(local_batch_size)).mean().item()
        steps += 1
        logger.info(f"(step={steps:04d}) [PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, "
                    f"LPIPS: {lpips:.4f}, FloLPIPS: {flolpips:.4f}, L1: {l1:.4f}")
        if args.save_generated_frames:
            if accelerator.is_local_main_process:
                blended_input = frame_0 * 0.5 + frame_1 * 0.5
                for sample_idx, (blended_sample, gt_sample, generated_sample) in enumerate(
                    zip(blended_input, gt, generated_frames)
                ):
                    file_name = build_visualization_name(dataset, saved_triplet_index + sample_idx)
                    save_image(blended_sample, f"{blended_dir}/{file_name}")
                    save_image(gt_sample, f"{gt_dir}/{file_name}")
                    save_image(generated_sample, f"{generated_dir}/{file_name}")
                saved_triplet_index += len(generated_frames)
                logger.info(f"Saved visualization results to {visualization_dir}")

    for key in results.keys():
        results[key] /= steps
    format_results = (f"PSNR: {results['PSNR']:.4f},  SSIM: {results['SSIM']:.4f}, LPIPS: {results['LPIPS']:.4f}, "
                      f"FloLPIPS: {results['FloLPIPS']:.4f}, L1: {results['L1']:.4f}")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        with open(f"{evaluation_dir}/evaluation_results.txt", mode="w+", encoding="utf-8") as f:
            f.write(format_results)
    logger.info(f"Pretrained {model_name} (ckpt:{args.pretrained_eden_path}) evaluation results on {dataset_name}: {format_results}")
    accelerator.end_training()
    model.eval()
    logger.info("Done!")


if __name__ == "__main__":
    main()

