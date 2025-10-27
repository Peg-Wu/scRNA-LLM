import sys
sys.path.append("../")

import argparse

from stella.tokenizer import WangLabTranscriptomeTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_qc", type=bool, default=False)
    parser.add_argument("--do_normalize", type=bool, default=True)
    parser.add_argument("--nproc", "-p", type=int, default=8)
    parser.add_argument("--max_length", "-l", type=int)
    parser.add_argument("--nbins", "-b", type=int)
    parser.add_argument("--save_dir", "-o", type=str)
    args = parser.parse_args()

    tokenizer = WangLabTranscriptomeTokenizer(
        seed=args.seed,
        nproc=args.nproc,
        max_length=args.max_length,
        bin_boundary_file=f"/fse/home/wupengpeng/STELLA/src/stella/bin_{args.nbins}.pkl",
        do_qc=args.do_qc,  # Pretrain data already did qc.
        do_normalize=args.do_normalize,
        custom_attr_name_dict=None
    )

    tokenizer(save_dir=args.save_dir)


if __name__ == "__main__":
    # nbins: 50; max_length: 2048
    # python w04_tokenize_data.py -p 8 -l 2048 -b 50 -o /fse/DC/stella/Human_B50_L2048

    # nbins: 50; max_length: 4096
    # python w04_tokenize_data.py -p 8 -l 4096 -b 50 -o /fse/DC/stella/Human_B50_L4096

    # nbins: 100; max_length: 2048
    # python w04_tokenize_data.py -p 8 -l 2048 -b 100 -o /fse/DC/stella/Human_B100_L2048

    # nbins: 100; max_length: 4096
    # python w04_tokenize_data.py -p 8 -l 4096 -b 100 -o /fse/DC/stella/Human_B100_L4096
    main()