import sys
sys.path.append("../")

from stella.data import GlobalBinExpression


def main():
    gbin = GlobalBinExpression(
        train_data_path="/fse/home/wupengpeng/STELLA/src/stella/train_data_path.pkl",
        nbins=50,
        nproc=32
    )

    gbin.process()
    gbin.save_bin_results("/fse/home/wupengpeng/STELLA/src/stella")


if __name__ == "__main__":
    main()