import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FileDataset(Dataset):
    def __init__(self, fnames):
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        return self.fnames[idx]

    def __len__(self):
        return self.num_samples


def get_dataset(data_dir):
    fnames = glob.glob(f"{data_dir}/*.jpg")
    dataset = FileDataset(fnames)
    return dataset


def predict(
    source, weights, img_size, thres, list_negative, progress, quiet, save_raw=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # workaround for namespace collision
    # ref: https://github.com/pytorch/hub/issues/243#issuecomment-942403391
    sys_modules_utils_value = sys.modules.pop("utils", None)

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weights,
        verbose=not quiet,
        _verbose=not quiet,
    ).to(device)

    if sys_modules_utils_value is not None:
        sys.modules["utils"] = sys_modules_utils_value

    model.eval()
    positive = 0
    nagative_list = []

    dataset = get_dataset(source)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=False)

    print(
        f"Found {len(dataset)} images in {Path(source).resolve().relative_to(Path.cwd())}"
    )

    print(f"Face detection threshold: {thres}")

    raw_results = []

    if progress and not quiet:
        dataloader = tqdm(dataloader)

    with torch.no_grad():
        for images in dataloader:
            outputs = model(images, size=img_size)

            for i, pred in enumerate(outputs.pred):
                if torch.sum(pred[:, 4] > thres) > 0:
                    positive += 1
                else:
                    nagative_list.append(outputs.files[i])

                if save_raw:
                    raw_results.append(
                        pred[0].cpu().numpy() if pred.shape[0] > 0 else np.zeros(6)
                    )

    results = {
        "positive": positive,
        "negative": len(dataset) - positive,
        "total": len(dataset),
        "positive_rate": positive / len(dataset),
        "negative_list": sorted(
            nagative_list, key=lambda f: int("".join(filter(str.isdigit, f)))
        ),
    }

    print(
        f"Positive rate: {results['positive']} / {results['total']} = {results['positive_rate']:.4f}"
    )

    if save_raw:
        np.save(
            Path(source).resolve().joinpath("raw_results.npy"), np.asarray(raw_results)
        )

    if list_negative:
        print("Negative list:")
        print("\n".join([f"\t{f}" for f in results["negative_list"]]))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="source")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "-p", "--progress", action="store_true", help="show progress bar"
    )
    # weights obtained from project https://github.com/zymk9/yolov5_anime
    # Downloaded from https://drive.google.com/file/d/1-MO9RYPZxnBfpNiGY6GdsqCeQWYNxBdl/view?usp=sharing
    parser.add_argument(
        "--weights", type=str, default="./yolov5x_anime.pt", help="model.pt path"
    )
    parser.add_argument("--thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument(
        "--img-size", type=int, default=64, help="inference size (pixels)"
    )
    parser.add_argument(
        "--list-negative", action="store_true", help="list negative images"
    )
    parser.add_argument(
        "--save-raw", action="store_true", help="save raw results as numpy array"
    )
    predict(**vars(parser.parse_args()))
