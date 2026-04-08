from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler
import av
import numpy as np
import torchvision
import torch
import argparse
import yaml
import os
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


def iter_video_frames(video_path):
    with av.open(video_path, metadata_errors="ignore") as container:
        for frame in container.decode(video=0):
            yield frame.to_rgb().to_ndarray()


def probe_video(video_path):
    with av.open(video_path, metadata_errors="ignore") as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")

        stream = container.streams.video[0]
        fps = stream.average_rate
        frame_count = 0
        frame_shape = None

        for frame in container.decode(video=0):
            if frame_shape is None:
                frame_shape = frame.to_rgb().to_ndarray().shape
            frame_count += 1

    if frame_shape is None:
        raise RuntimeError(f"No video frames found in {video_path}")
    if fps is None:
        raise RuntimeError(f"Could not determine FPS for {video_path}")

    return float(fps), frame_count, frame_shape


def frame_array_to_tensor(frame_array):
    return torch.from_numpy(frame_array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.)


def tensor_to_video_frame(frame_tensor):
    return frame_tensor.squeeze(0).permute(1, 2, 0).mul(255.).clamp(0., 255.).to(torch.uint8).cpu().numpy()


device = "cuda:0"
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/eval_eden.yaml")
parser.add_argument("--frame_0_path", type=str, default="examples/frame_0.jpg")
parser.add_argument("--frame_1_path", type=str, default="examples/frame_1.jpg")
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--interpolated_results_dir", type=str, default="interpolation_outputs")
parser.add_argument("--video_codec", type=str, default="libx264rgb")
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
    video_write_options = {"crf": "0"}
    fps, frames_num, frame_shape = probe_video(video_path)
    output_frames_num = max(2 * frames_num - 1, 1)
    with tempfile.TemporaryDirectory(prefix="eden_inference_") as temp_dir:
        memmap_path = os.path.join(temp_dir, "interpolated_video.dat")
        interpolated_video = np.memmap(memmap_path, dtype=np.uint8, mode="w+",
                                       shape=(output_frames_num, *frame_shape))
        frame_iterator = iter_video_frames(video_path)
        previous_frame = next(frame_iterator)
        interpolated_video[0] = previous_frame
        output_index = 1

        for next_frame in frame_iterator:
            with torch.no_grad():
                frame_0 = frame_array_to_tensor(previous_frame)
                frame_1 = frame_array_to_tensor(next_frame)
                interpolated_frame = interpolate(frame_0, frame_1)
            interpolated_video[output_index] = tensor_to_video_frame(interpolated_frame)
            interpolated_video[output_index + 1] = next_frame
            previous_frame = next_frame
            output_index += 2
            del frame_0, frame_1, interpolated_frame
            torch.cuda.empty_cache()

        interpolated_video.flush()
        torchvision.io.write_video(
            interpolated_video_save_path,
            interpolated_video,
            fps=2 * fps,
            video_codec=args.video_codec,
            options=video_write_options,
        )
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

