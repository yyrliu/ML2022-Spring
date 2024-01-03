import argparse
import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class FileDataset(Dataset):
    def __init__(self, fnames):
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        return self.fnames[idx]

    def __len__(self):
        return self.num_samples


def get_dataset(data_dir):
    fnames = glob.glob(f"{data_dir}/*")
    dataset = FileDataset(fnames)
    return dataset


def predict(source, weights, img_size, thres, list_negative, pbar, quiet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weights,
        verbose=not quiet,
        _verbose=not quiet
    ).to(device)

    model.eval()
    positive = 0
    nagative_list = []

    dataset = get_dataset(source)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=False)

    print(f"Found {len(dataset)} images in {Path(source).resolve()}")

    if pbar and not quiet:
        dataloader = tqdm(dataloader)

    with torch.no_grad():
        for images in dataloader:
            outputs = model(images, size=img_size)

            for i, pred in enumerate(outputs.pred):
                if torch.sum(pred[:, 4] > thres) > 0:
                    positive += 1
                else:
                    nagative_list.append(outputs.files[i])

    print(f"Positive rate: {positive} / {len(dataset)} = {positive / len(dataset)}")
    if list_negative:
        print("Negative list:")
        for f in sorted(nagative_list, key=lambda f: int(''.join(filter(str.isdigit, f)))):
            print(f"\t{f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source')
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument('--pbar', action='store_true', help='show progress bar')
    # weights obtained from project https://github.com/zymk9/yolov5_anime
    # Downloaded from https://drive.google.com/file/d/1-MO9RYPZxnBfpNiGY6GdsqCeQWYNxBdl/view?usp=sharing
    parser.add_argument('--weights', type=str, default='./yolov5x_anime.pt', help='model.pt path')
    parser.add_argument('--thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--img-size', type=int, default=64, help='inference size (pixels)')
    parser.add_argument('--list-negative', action='store_true', help='list negative images')
    predict(**vars(parser.parse_args()))
