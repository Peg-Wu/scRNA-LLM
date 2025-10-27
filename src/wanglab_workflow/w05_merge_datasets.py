import sys
sys.path.append("../")

import os
import argparse
from scphd.data import merge_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str)
    args = parser.parse_args()

    save_dir = os.path.join(args.data_dir, "merge")
    merge_datasets(data_dir=args.data_dir, save_dir=save_dir, chunk_size=1024, nproc=16)


if __name__ == "__main__":
    # nbins: 50; max_length: 2048
    # python w05_merge_datasets.py -d /fse/DC/stella/Human_B50_L2048

    # nbins: 50; max_length: 4096
    # python w05_merge_datasets.py -d /fse/DC/stella/Human_B50_L4096

    # nbins: 100; max_length: 2048
    # python w05_merge_datasets.py -d /fse/DC/stella/Human_B100_L2048

    # nbins: 100; max_length: 4096
    # python w05_merge_datasets.py -d /fse/DC/stella/Human_B100_L4096
    main()
