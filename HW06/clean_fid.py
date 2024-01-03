import argparse
import glob
from collections import defaultdict
from functools import reduce
from pathlib import Path

from cleanfid import fid


def eval(dir, quiet=False):
    if isinstance(dir, Path):
        dir = str(dir)

    valid_dirs = {
        "crypko": "./data/faces",
        "animeface": "./data/images",
        "animeface_2000": "./data/validate_set",
    }

    print(
        f"Found {len(glob.glob(f'{dir}/*.jpg'))} images in {Path(dir).resolve().relative_to(Path.cwd())}"
    )

    results = defaultdict(dict)

    for dataset_name, dataset_path in valid_dirs.items():
        # Check if a custom statistic already exists
        if not fid.test_stats_exists(dataset_name, mode="clean"):
            # Generating custom statistics (saved to local cache)
            fid.make_custom_stats(dataset_name, dataset_path, mode="clean")

        # Using the generated custom statistics
        fid_score = fid.compute_fid(
            dir,
            dataset_name=dataset_name,
            mode="clean",
            dataset_split="custom",
            verbose=not quiet,
        )

        kid_score = fid.compute_kid(
            dir,
            dataset_name=dataset_name,
            mode="clean",
            dataset_split="custom",
            verbose=not quiet,
        )

        results[dataset_name]["fid"] = fid_score
        results[dataset_name]["kid"] = kid_score

    # mock results
    # results = {
    #     "crypko": {"fid": 1.0, "kid": 2.0},
    #     "animeface": {"fid": 3.0, "kid": 4.0},
    #     "animeface_1000": {"fid": 5.0, "kid": 6.0},
    # }

    for key, fid_score in map(lambda kv: (kv[0], kv[1]["fid"]), results.items()):
        print(f"FID - {key:<15} {fid_score:>10.1f}")

    for key, kid_score in map(lambda kv: (kv[0], kv[1]["kid"]), results.items()):
        print(f"KID - {key:<15} {kid_score:>10.4f}")

    results["Average"] = {
        "fid": reduce(lambda r, v: v["fid"] + r, results.values(), 0) / len(results),
        "kid": reduce(lambda r, v: v["kid"] + r, results.values(), 0) / len(results),
    }

    print(f"FID - {'Average':<15} {results['Average']['fid']:>10.1f}")
    print(f"KID - {'Average':<15} {results['Average']['fid']:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    eval(**vars(args))
