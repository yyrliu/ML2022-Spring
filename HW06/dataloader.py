import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# prepare for CrypkoDataset
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(data_dir):
    logger.info(f"Loading dataset from {data_dir}")
    fnames = glob.glob(f"{data_dir}/*")
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    logger.info(f"Dataset loaded, size: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    workspace_dir = "./data"
    temp_dataset = get_dataset(Path(workspace_dir, "faces"))
    images = [temp_dataset[i] for i in range(4)]
    print(images[0].shape)
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
