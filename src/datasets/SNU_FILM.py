from pathlib import Path

import torch, torchvision
from torch.utils.data import Dataset
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore")


class SNUFILMDataset(Dataset):
    def __init__(self, data_dir="datasets/SNU_FILM", mode="extreme"):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.data_root = self.resolve_data_root(data_dir)
        self.mode = mode
        self.meta_data = self.read_split_file(mode)

    def __len__(self):
        return len(self.meta_data)

    def resolve_data_root(self, data_dir):
        data_dir = Path(data_dir)
        candidates = [
            data_dir,
            self.repo_root / data_dir,
            self.repo_root / "data" / "SNU-FILM",
            self.repo_root / "datasets" / "SNU_FILM",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"SNU-FILM dataset directory not found. Tried: {', '.join(str(path) for path in candidates)}"
        )

    def read_split_file(self, mode):
        split_candidates = [
            self.data_root / "eval_modes" / f"test-{mode}.txt",
            self.data_root / f"test-{mode}.txt",
        ]
        split_file = next((path for path in split_candidates if path.exists()), None)
        if split_file is None:
            raise FileNotFoundError(
                f"SNU-FILM split file not found for mode '{mode}'. Tried: {', '.join(str(path) for path in split_candidates)}"
            )

        triplets = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                frame_paths = tuple(self.resolve_frame_path(path_str) for path_str in line.split())
                if len(frame_paths) != 3:
                    raise ValueError(f"Expected 3 frame paths per line in {split_file}, got {len(frame_paths)}: {line}")
                triplets.append(frame_paths)
        return triplets

    def resolve_frame_path(self, path_str):
        raw_path = Path(path_str)
        candidates = [
            raw_path,
            self.repo_root / raw_path,
            self.data_root / raw_path,
        ]

        raw_parts = raw_path.parts
        if "test" in raw_parts:
            test_index = raw_parts.index("test")
            candidates.append(self.data_root / Path(*raw_parts[test_index:]))
        elif raw_parts:
            candidates.append(self.data_root / "test" / raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"SNU-FILM frame not found for '{path_str}'. Tried: {', '.join(str(path) for path in candidates)}"
        )

    def __getitem__(self, index):
        frame0_path, gt_path, frame1_path = self.meta_data[index]
        frame0 = torchvision.io.read_image(str(frame0_path))
        frame1 = torchvision.io.read_image(str(frame1_path))
        gt = torchvision.io.read_image(str(gt_path))
        frames = torch.stack((frame0, frame1, gt), dim=0)
        return frames
