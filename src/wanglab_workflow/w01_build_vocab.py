import sys
sys.path.append("../")

from stella.vocab import WangLabGeneSymbolVocabBuilder


SOURCE_DATA_PATH = [
    "/fse/DC/human/qc_final_tmp",          # round 1
    "/fse/DC/human/second_qc_final",       # round 2
]


def main():
    vocab_builder = WangLabGeneSymbolVocabBuilder(nproc=32)
    vocab_builder(SOURCE_DATA_PATH, min_freq=0, vocab_save_dir="/fse/home/wupengpeng/STELLA/src/stella")


if __name__ == "__main__":
    main()