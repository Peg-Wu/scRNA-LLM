import pickle
import logging
import scanpy as sc

from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, List
from collections import Counter
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)

# Pretrain task only involve reconstruction now!
STELLA_SPECIAL_TOKENS = {
    "[PAD]": 0
}

PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"

PAD_TOKEN_ID = 0  # both in gene_symbol_vocab abd bin_vocab!
MASK_TOKEN_ID = 101  # To be the last token in bin_vocab!


class GeneSymbolVocabBuilder:
    def __init__(self, nproc: int = 1):
        r"""
        **Parameters:**
        nproc : int
            | Number of processes.
        """
        self.vocab = {} | STELLA_SPECIAL_TOKENS
        self.num_special_tokens = len(STELLA_SPECIAL_TOKENS)
        self.gene_counter = Counter()
        self.nproc = nproc

    def find_h5ad(self, filepath: Union[Path, str]) -> List[Path]:
        if not isinstance(filepath, Path):
            filepath = Path(filepath).absolute()

        fp = list(filepath.rglob("*.h5ad"))

        logger.info(f"Find {len(fp)} h5ad files!")
        return fp

    def extracted_from_h5ad(self, h5ad_fp: Path) -> Counter:
        # you can modify here to extract gene symbols from your own h5ad file.
        adata = sc.read_h5ad(h5ad_fp, backed="r")
        adata_genes = adata.var_names.str.replace(r'\.\d+$', '', regex=True)
        return Counter(adata_genes)

    def remove_low_freq_genes(self, gene_counter: Counter, min_freq: int) -> List[str]:
        final_gene_list = [k for k, v in gene_counter.items() if v > min_freq]

        logger.info(f"Totally remove {len(gene_counter) - len(final_gene_list)} genes.")
        return final_gene_list

    def _build(self, filepath: Union[Path, str]):
        
        logger.info(f"Processing h5ad files in {str(filepath)}")

        fp = self.find_h5ad(filepath)

        counters = Parallel(n_jobs=self.nproc)(delayed(self.extracted_from_h5ad)(f) for f in tqdm(fp))
        for counter in counters:
            self.gene_counter += counter

    def __call__(
        self, 
        h5ad_dirs: Union[List[str], str],
        min_freq: int = 3,
        vocab_save_dir: Union[Path, str] = "./"
    ):
        r"""
        Build a gene dictionary from all h5ad files in h5ad_dirs.

        **Examples:**
            >>> from stella.vocab import GeneSymbolVocabBuilder
            >>> gvb = GeneSymbolVocabBuilder(nproc=8)
            
            >>> data = ["dir1", "dir2", ...]
            >>> gvb(data, min_freq=0, vocab_save_dir="./")

        **Parameters:**
        h5ad_dirs : Union[List[str], str]
            | H5ad file folders.
        min_freq : int
            | If the frequency of the gene is min_freq or less, it will be removed.
        vocab_save_dir : Union[Path, str]
            | Vocab save directory.
        """
        if min_freq != 0 and "[UNK]" not in STELLA_SPECIAL_TOKENS:
            raise ValueError("If you want to remove low-frequency genes, please use '[UNK]' token!")

        if isinstance(h5ad_dirs, str):
            h5ad_dirs = [h5ad_dirs]
        
        for h5ad_dir in h5ad_dirs:
            self._build(h5ad_dir)
        
        final_gene_list = self.remove_low_freq_genes(self.gene_counter, min_freq)

        self.vocab = \
            self.vocab | dict(zip(final_gene_list, range(self.num_special_tokens, len(final_gene_list) + self.num_special_tokens)))

        logger.info(f"Vocab size: {len(self.vocab)}, including {self.num_special_tokens} special tokens.")

        # save vocab
        with open(Path(vocab_save_dir) / "gene2id.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        
        logger.info(f"Saving vocab to {str(Path(vocab_save_dir).absolute())}")


###################
##### WangLab #####
###################

class WangLabGeneSymbolVocabBuilder(GeneSymbolVocabBuilder):

    r"""
    1. 第一轮:
        - h5ad: "/fse/DC/human/qc_final_tmp"
        - meta: "/fse/DC/human/meta_info"

    2. 第二轮:
        - h5ad: "/fse/DC/human/second_qc_final"
        - meta: "/fse/DC/human/second_meta_info"

    3. 训练时只使用有 raw_counts 的 h5ad 文件, 取 raw_counts 的方法: adata.raw.to_adata()

    4. 由于数据提供者可能运行了 var_names_make_unique, 因此应该移除基因 symbol 后面的.1, .2, ...

    5. reserve.txt 和 normalize_scale.txt 文件的含义:
        - reserve.txt: raw_counts + normalize (w/o QC)
        - normalize_scale.txt: normalize (w/o QC) + normalize (w QC)
        - 所以 reserve.txt 和 normalize_scale.txt 文件中的路径做差集就是最终要处理的文件
    """

    def extracted_from_h5ad(self, h5ad_fp: Path) -> Counter:
        # step1: 判断 h5ad_fp 是不是 raw_counts, 是: process, 不是: skip
        if h5ad_fp not in self.raw_counts_fp:
            return Counter()
        else:
            # step2: 将 adata 中的 gene 名称提取出来, 并将名称中的.1, .2, ...移除
            adata = sc.read_h5ad(h5ad_fp, backed="r")
            adata_genes = adata.raw.var_names.str.replace(r'\.\d+$', '', regex=True)

            # step3: 返回 Counter 对象统计每个 gene 名称出现的频次
            return Counter(adata_genes)

    def add_attr_raw_counts_fp(self, h5ad_dir: Union[Path, str]):
        if not isinstance(h5ad_dir, Path):
            h5ad_dir = Path(h5ad_dir)
        
        # 利用 reserve.txt 和 normalize_scale.txt 两个文件提取 raw_counts 文件路径
        fr_path_list, fn_path_list = [], []
        with open(h5ad_dir / "reserve.txt", "r") as fr, open(h5ad_dir / "normalize_scale.txt", "r") as fn:
            for line in fr:
                p = Path(line.strip().split(",")[0])
                p = p.with_suffix(".h5ad")
                suffix = p.parts[5:]
                prefix = Path(h5ad_dir).parts
                p = Path(*(prefix + suffix))
                fr_path_list.append(p)
            
            for line in fn:
                p = Path(line.strip())
                p = p.with_suffix(".h5ad")
                suffix = p.parts[5:]
                prefix = Path(h5ad_dir).parts
                p = Path(*(prefix + suffix))
                fn_path_list.append(p)

        self.raw_counts_fp = list(set(fr_path_list) - set(fn_path_list))

    def __call__(
        self, 
        h5ad_dirs: Union[List[str], str],
        min_freq: int = 3,
        vocab_save_dir: Union[Path, str] = "./"
    ):
        r"""
        **Usage:**
            >>> from stella.vocab import WangLabGeneSymbolVocabBuilder
            >>> data = [
            ...     "/fse/DC/human/qc_final_tmp",          # round 1
            ...     "/fse/DC/human/second_qc_final",       # round 2
            ... ]
            >>> gvb = WangLabGeneSymbolVocabBuilder(nproc=32)
            >>> gvb(data, min_freq=0, vocab_save_dir="./stella")
        """
        if min_freq != 0 and "[UNK]" not in STELLA_SPECIAL_TOKENS:
            raise ValueError("If you want to remove low-frequency genes, please use '[UNK]' token!")

        if isinstance(h5ad_dirs, str):
            h5ad_dirs = [h5ad_dirs]
        
        for h5ad_dir in h5ad_dirs:
            self.add_attr_raw_counts_fp(h5ad_dir)
            self._build(h5ad_dir)

        final_gene_list = self.remove_low_freq_genes(self.gene_counter, min_freq)

        self.vocab = \
            self.vocab | dict(zip(final_gene_list, range(self.num_special_tokens, len(final_gene_list) + self.num_special_tokens)))

        logger.info(f"Vocab size: {len(self.vocab)}, including {self.num_special_tokens} special tokens.")

        # save gene2id & gene2freq
        with open(Path(vocab_save_dir) / "gene2id.pkl", "wb") as f1, open(Path(vocab_save_dir) / "gene2freq.pkl", "wb") as f2:
            pickle.dump(self.vocab, f1)
            pickle.dump(dict(self.gene_counter), f2)
        
        logger.info(f"Saving vocab to {str(Path(vocab_save_dir).absolute())}")


if __name__ == "__main__":
    pass
