import argparse
import glob
import logging
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import torch
import torchvision

import clean_fid
import config as cfg
from model import Generator
from utils import comfirm_overwrite, fix_random_seed, load_config, setup_logger
from yolov5_anime import predict as detect_faces

logger = logging.getLogger(__name__)


def inference_during_train(
    generator, step, device, stats, save_dir=None, n_generate=1000
):
    if save_dir is None:
        save_dir = Path(cfg.config.workspace_dir, f"temp_{step}")

    try:
        Path(save_dir).mkdir()
    except FileExistsError:
        rmtree(save_dir)
        Path(save_dir).mkdir()

    logger.info(f'Temporary dir "{save_dir}" created for inference during training.')

    generator.eval()
    z_samples = torch.randn(n_generate, cfg.config.z_dim).to(device)
    with torch.no_grad():
        imgs = (generator(z_samples) + 1) / 2.0

    for i, img in enumerate(imgs):
        torchvision.utils.save_image(img, Path(save_dir, f"{i:03}.jpg"))

    fid_results, face_results = eval(save_dir, fid_valid_sets=cfg.config.valid_fid_set)

    current_stats = {
        "gen/fid": fid_results[cfg.config.valid_fid_set]["fid"],
        "gen/kid": fid_results[cfg.config.valid_fid_set]["kid"],
        "gen/afd": face_results["positive_rate"],
    }

    for key, value in current_stats.items():
        stats[key].append(value)

    rmtree(save_dir)
    logger.info(f'Temporary dir "{save_dir}" removed.')

    return stats


def inference(
    checkpoint=None, n_generate=1000, n_output=100, view=False, overwrite=False
):
    """
    1. G_path is the path for Generator ckpt
    2. You can use this function to generate final answer
    """

    logger = setup_logger("hw6.gan", use_wandb=False)

    output_dir = Path(cfg.config.workspace_dir, "output")

    comfirm_overwrite(f"{cfg.config.workspace_dir}/output/*", overwrite)
    Path(output_dir).mkdir(exist_ok=True)

    if checkpoint is None:
        checkpoint = sorted(
            Path(cfg.config.ckpt_dir).glob("G_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
            reverse=True,
        )[0]

    logger.info(f"Loading checkpoint from {checkpoint}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator(cfg.config.z_dim).to(device)
    generator.load_state_dict(torch.load(checkpoint))
    generator.eval()
    z_samples = torch.randn(n_generate, cfg.config.z_dim).to(device)
    with torch.no_grad():
        imgs = (generator(z_samples) + 1) / 2.0

    for i, img in enumerate(imgs):
        torchvision.utils.save_image(img, Path(output_dir, f"{i:03}.jpg"))

    if view:
        row, col = n_output // 10, 10
        grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
        plt.figure(figsize=(row, col))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()


def eval(
    dir, save_to=None, overwrite=False, fid_valid_sets=None, face_detection_thres=None
):
    if face_detection_thres is None:
        face_detection_thres = cfg.config.valid_afd_thres
    fid_results = clean_fid.eval(dir, valid_sets=fid_valid_sets)
    face_results = detect_faces(
        dir,
        weights="./yolov5x_anime.pt",
        img_size=64,
        thres=face_detection_thres,
        list_negative=False,
        progress=True,
        quiet=False,
    )

    if save_to is not None:
        comfirm_overwrite(save_to, overwrite)

        with open(save_to, "w") as f:
            f.write(
                f"Found {len(glob.glob(f'{dir}/*.jpg'))} images in {Path(dir).resolve().relative_to(Path.cwd())}\n"
            )

            f.write("\n---------- FID/KID ----------\n")

            for key, fid_score in map(
                lambda kv: (kv[0], kv[1]["fid"]), fid_results.items()
            ):
                f.write(f"FID - {key:<15} {fid_score:>10.1f}\n")

            for key, kid_score in map(
                lambda kv: (kv[0], kv[1]["kid"]), fid_results.items()
            ):
                f.write(f"KID - {key:<15} {kid_score:>10.4f}\n")

            f.write("\n---------- Face Detection ----------\n")

            f.write(f"Face detection threshold: {face_detection_thres}\n")

            f.write(
                f"Positive rate: {face_results['positive']} / {face_results['total']} = {face_results['positive_rate']:.4f}\n"
            )

            f.write("Negative list:\n")
            f.write("\n".join([f"\t{f}" for f in face_results["negative_list"]]))

        print(f"Result saved to {save_to}")

    return fid_results, face_results


if __name__ == "__main__":
    fix_random_seed(2022)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "-n", "--samples", help="Number of samples to generate", type=int, default=1000
    )
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-v", "--view", action="store_true")
    parser.add_argument(
        "--model", help="Path to Generator checkpoint", type=str, default=None
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
    inference(
        checkpoint=args.model,
        n_generate=args.samples,
        view=args.view,
        overwrite=args.force,
    )

    if not args.no_eval:
        dir_path = Path(cfg.config.workspace_dir, "output").resolve()
        print(f"Evaluating generated images in {dir_path}")
        eval(
            dir_path,
            save_to=dir_path.parent.joinpath("result.txt")
            if args.save_result
            else None,
            overwrite=args.force,
        )
