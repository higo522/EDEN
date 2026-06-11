from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler
from fractions import Fraction
from pathlib import Path
import torchvision
import torch
import argparse
import yaml
import os
import shutil
import subprocess
import tempfile


def interpolate(frame0, frame1):
    h, w = frame0.shape[2:]
    image_size = [h, w]
    padder = InputPadder(image_size)
    difference = ((torch.mean(torch.cosine_similarity(frame0, frame1),
                              dim=[1, 2]) - args.cos_sim_mean) / args.cos_sim_std).unsqueeze(1).to(device)
    cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
    new_h, new_w = cond_frames.shape[2:]
    noise = torch.randn([1, new_h // 32 * new_w // 32, args.model_args["latent_dim"]]).to(device)
    denoise_kwargs = {"cond_frames": cond_frames, "difference": difference}
    samples = sample_fn(noise, eden.denoise, **denoise_kwargs)[-1]
    denoise_latents = samples / args.vae_scaler + args.vae_shift
    generated_frame = eden.decode(denoise_latents)
    generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
    return generated_frame


def get_video_fps(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(Fraction(result.stdout.strip()))


def extract_video_frames(video_path, frames_dir):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vsync",
            "0",
            str(Path(frames_dir) / "frame_%08d.png"),
        ],
        check=True,
    )


def encode_video_frames(frames_dir, output_path, fps):
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(Path(frames_dir) / "frame_%08d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ],
        check=True,
    )


device = "cuda:0"
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/eval_eden.yaml")
parser.add_argument("--frame_0_path", type=str, default="examples/frame_0.jpg")
parser.add_argument("--frame_1_path", type=str, default="examples/frame_1.jpg")
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--interpolated_results_dir", type=str, default="interpolation_outputs")
args = parser.parse_args()
with open(args.config, "r") as f:
    update_args = yaml.unsafe_load(f)
parser.set_defaults(**update_args)
args = parser.parse_args()
model_name = args.model_name
eden = load_model(model_name, **args.model_args)
ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")
eden.load_state_dict(ckpt["eden"])
eden.to(device)
eden.eval()
del ckpt
transport = create_transport("Linear", "velocity")
sampler = Sampler(transport)
sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
video_path = args.video_path
interpolated_results_dir = args.interpolated_results_dir
os.makedirs(interpolated_results_dir, exist_ok=True)
frame_0_path, frame_1_path = args.frame_0_path, args.frame_1_path
if video_path:
    print(f"Interpolating Video ({video_path}) ...")
    interpolated_video_save_path = f"{interpolated_results_dir}/interpolated.mp4"
    fps = get_video_fps(video_path)
    with tempfile.TemporaryDirectory(prefix="eden_video_in_") as input_frames_dir, \
            tempfile.TemporaryDirectory(prefix="eden_video_out_") as output_frames_dir:
        extract_video_frames(video_path, input_frames_dir)
        input_frame_paths = sorted(Path(input_frames_dir).glob("frame_*.png"))

        if len(input_frame_paths) < 2:
            raise RuntimeError(f"Expected at least 2 frames in {video_path}, found {len(input_frame_paths)}")

        output_index = 1
        shutil.copyfile(input_frame_paths[0], Path(output_frames_dir) / f"frame_{output_index:08d}.png")
        output_index += 1

        for pair_index, (frame_0_file, frame_1_file) in enumerate(zip(input_frame_paths, input_frame_paths[1:]), start=1):
            with torch.no_grad():
                frame_0 = (torchvision.io.read_image(str(frame_0_file)) / 255.).unsqueeze(0).to(device)
                frame_1 = (torchvision.io.read_image(str(frame_1_file)) / 255.).unsqueeze(0).to(device)
                interpolated_frame = interpolate(frame_0, frame_1)

            torchvision.utils.save_image(
                interpolated_frame,
                str(Path(output_frames_dir) / f"frame_{output_index:08d}.png"),
            )
            output_index += 1
            shutil.copyfile(frame_1_file, Path(output_frames_dir) / f"frame_{output_index:08d}.png")
            output_index += 1
            del frame_0, frame_1, interpolated_frame
            torch.cuda.empty_cache()

            if pair_index % 50 == 0:
                print(f"Processed {pair_index}/{len(input_frame_paths) - 1} frame pairs...")

        encode_video_frames(output_frames_dir, interpolated_video_save_path, fps * 2)
    print(f"Saved interpolated video in {interpolated_video_save_path}.")
elif frame_0_path and frame_1_path:
    print(f"Interpolating Image-pairs {frame_0_path}-{frame_1_path} ...")
    frame_0 = (torchvision.io.read_image(frame_0_path) / 255.).unsqueeze(0).to(device)
    frame_1 = (torchvision.io.read_image(frame_1_path) / 255.).unsqueeze(0).to(device)
    interpolated_frame = interpolate(frame_0, frame_1)
    interpolated_frame_path = f"{interpolated_results_dir}/interpolated.png"
    torchvision.utils.save_image(interpolated_frame, interpolated_frame_path)
    print(f"Saved interpolated image in {interpolated_frame_path}.")
else:
    assert "There are no images or videos to be interpolated!"