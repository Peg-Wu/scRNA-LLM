import sys
sys.path.append("../")

from stella.data import WangLabTrainDataPathExtractor


SOURCE_DATA_PATH = [
    "/fse/DC/human/qc_final_tmp",          # round 1
    "/fse/DC/human/second_qc_final",       # round 2
]


def main():
    extractor = WangLabTrainDataPathExtractor(nproc=32)
    extractor(SOURCE_DATA_PATH, save_dir="/fse/home/wupengpeng/STELLA/src/stella", count_total_train_cells=True)


if __name__ == "__main__":
    main()