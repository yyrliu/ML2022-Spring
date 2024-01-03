import argparse
from collections import defaultdict
from functools import reduce
from pathlib import Path

from cleanfid import fid


def write_results(results, file):
    with open(file, "w") as f:
        f.write('FID/KID for "./exps/baseline/output"\n')
        f.write("Number of images: 1000\n")
        f.write("---------------------------------------\n")
        for key, fid in map(lambda kv: (kv[0], kv[1]["fid"]), results.items()):
            f.write(f"FID - {key:<15} {fid:>10.1f}\n")
        f.write("---------------------------------------\n")
        for key, kid in map(lambda kv: (kv[0], kv[1]["kid"]), results.items()):
            f.write(f"KID - {key:<15} {kid:>10.4f}\n")
        f.write("---------------------------------------\n")
        f.write(f"FID - {'Average':<15} {reduce(lambda r, v: v['fid'] + r, results.values(), 0) / len(results):>10.1f}\n")
        f.write(f"KID - {'Average':<15} {reduce(lambda r, v: v['kid'] + r, results.values(), 0) / len(results):>10.4f}\n")


def eval(dir):
    valid_dirs = {
        "crypko": "./data/faces",
        "animeface": "./data/images",
        "animeface_2000": "./data/validate_set",
    }

    results = defaultdict(dict)

    for dataset_name, dataset_path in valid_dirs.items():
        # Check if a custom statistic already exists
        if not fid.test_stats_exists(dataset_name, mode="clean"):
            # Generating custom statistics (saved to local cache)
            fid.make_custom_stats(dataset_name, dataset_path, mode="clean")

        # Using the generated custom statistics
        fid_score = fid.compute_fid(
            dir, dataset_name=dataset_name, mode="clean", dataset_split="custom"
        )

        kid_score = fid.compute_kid(
            dir, dataset_name=dataset_name, mode="clean", dataset_split="custom"
        )

        results[dataset_name]["fid"] = fid_score
        results[dataset_name]["kid"] = kid_score

        print(f"FID: {dataset_name} - {dir}: {fid_score}")
        print(f"KID: {dataset_name} - {dir}: {kid_score}")

    # mock results
    # results = {
    #     "crypko": {"fid": 1.0, "kid": 2.0},
    #     "animeface": {"fid": 3.0, "kid": 4.0},
    #     "animeface_1000": {"fid": 5.0, "kid": 6.0},
    # }

    write_results(results, Path(dir).parent.joinpath("results.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    eval(args.dir)
