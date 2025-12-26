import shutil
import pickle
import logging
import numpy as np
import scanpy as sc

from pathlib import Path
from tqdm.auto import tqdm
from tdigest import TDigest
from joblib import Parallel, delayed
from typing import Union, Optional, List
from datasets import load_from_disk, concatenate_datasets

logger = logging.getLogger(__name__)


class TrainDataPathExtractor:
    def __init__(self, nproc: int = 1):
        r"""
        **Parameters:**
        nproc : int
            | Number of processes. (Accelerate the count of training cells)
        """
        self.train_data_path = {"data": []}
        self.nproc = nproc

    def find_h5ad(self, filepath: Union[Path, str]) -> List[Path]:
        if not isinstance(filepath, Path):
            filepath = Path(filepath).absolute()

        fp = list(filepath.rglob("*.h5ad"))

        logger.info(f"Find {len(fp)} h5ad files in {str(filepath)}!")
        return fp

    def count_train_cells_from_h5ad(self, h5ad_fp: Path) -> int:
        return len(sc.read_h5ad(h5ad_fp, backed="r"))

    def __call__(
        self, 
        h5ad_dirs: Union[List[str], str], 
        save_dir: Optional[Union[Path, str]] = None,
        count_total_train_cells: bool = False
    ):
        r"""
        Extract the path of all h5ad files in h5ad_dirs.

        **Examples:**
            >>> from stella.data import TrainDataPathExtractor
            >>> extractor = TrainDataPathExtractor(nproc=8)

            >>> data = ["dir1", "dir2", ...]
            >>> extractor(data, count_total_train_cells=True)

            >>> print(extractor.train_data_path["data"])         # list of training data path.
            >>> print(extractor.train_data_path["ncells"])       # number of training cells.

        **Parameters:**
        h5ad_dirs : Union[List[str], str]
            | H5ad file folders.
        save_dir : Union[Path, str]
            | Train data path save directory.
        count_total_train_cells : bool
            | Count the total number of cells in the training data set. (Warning: Time Consuming!)
        """
        if isinstance(h5ad_dirs, str):
            h5ad_dirs = [h5ad_dirs]
        
        for h5ad_dir in h5ad_dirs:
            self.train_data_path["data"].extend(self.find_h5ad(h5ad_dir))
        
        if count_total_train_cells:
            counts = Parallel(n_jobs=self.nproc)(delayed(self.count_train_cells_from_h5ad)(f) \
                                                 for f in tqdm(self.train_data_path["data"]))
            self.train_data_path["ncells"] = sum(counts)
            logger.info(f"Total training cells: {self.train_data_path['ncells']:,d}")

        # save
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(save_dir) / "train_data_path.pkl", "wb") as f:
                pickle.dump(self.train_data_path, f)
            logger.info(f"Train data path has saved into {str(save_dir)}!")


class GlobalBinExpression:
    r"""
    **Examples:**
        >>> from stella.data import GlobalBinExpression
        >>> gbin = GlobalBinExpression(train_data_path=..., nproc=32, nbins=50)
        >>> gbin.process()
        >>> gbin.save_bin_results("./")

        **Parameters:**
        train_data_path : str
            | Generate from `TrainDataPathExtractor`.
        nbins : int
            | Number of bins you want to split expression into.
    """
    def __init__(
        self,
        train_data_path: str,
        nbins: int = 50,
        nproc: int = 32
    ):
        with open(train_data_path, "rb") as f:
            self.train_data_path = pickle.load(f)["data"]
        
        self.nbins = nbins
        self.nproc = nproc

        self.global_tdigest = None  # generate by self.process
        self.bin_results = None  # generate by self.process

    def _process(self, fp: str) -> TDigest:
        tmp_tdigest = TDigest()

        # read adata raw counts
        adata = sc.read_h5ad(fp).raw.to_adata()  # WangLab raw counts location, please modify it!
        
        # normalize & log1p
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # extract nonzero values
        X = adata.X.toarray()
        X = X[X != 0]

        # update
        tmp_tdigest.batch_update(X)
        
        return tmp_tdigest

    def process(self):
        # reinitialization
        self.bin_results = []
        self.global_tdigest = TDigest()

        # process
        all_tdigests = \
            Parallel(n_jobs=self.nproc)(
                delayed(self._process)(fp) for fp in tqdm(self.train_data_path)
            )
        
        # merge all tdigests
        logger.info("Merging all tdigest objects ...")
        for tdigest in tqdm(all_tdigests):
            self.global_tdigest += tdigest
        
        # bin
        logger.info("Binning ...")
        for q in tqdm(np.linspace(0, 1, self.nbins + 1)):
            self.bin_results.append(self.global_tdigest.percentile(q * 100))

    def save_bin_results(self, save_dir: str):
        logger.info("Saving bin tdigest object ...")
        with open(Path(save_dir) / "tdigest.pkl", "wb") as f:
            pickle.dump(self.global_tdigest, f)  # You can reuse `global_tdigest` to quickly split bins!

        logger.info("Saving bin results ...")
        with open(Path(save_dir) / f"bin_{self.nbins}.pkl", "wb") as f:
            pickle.dump(self.bin_results, f)


def global_bin_from_tdigest_pkl_file(
    tdigest_pkl_file: str,
    nbins: int,
    save_dir: str
):
    logger.info(f"Bin into {nbins} ...")
    with open(tdigest_pkl_file, "rb") as f:
        global_tdigest = pickle.load(f)

    bin_results = []
    for q in tqdm(np.linspace(0, 1, nbins + 1)):
        bin_results.append(global_tdigest.percentile(q * 100))
    
    logger.info("Saving bin results ...")
    with open(Path(save_dir) / f"bin_{nbins}.pkl", "wb") as f:
        pickle.dump(bin_results, f)


def merge_datasets(
    data_dir: str,
    save_dir: str,
    nproc: int = 16,
    chunk_size: int = 1024,
    max_shard_size: str = "5GB"
):
    """Storage required: dataset size * 3, this function need to be optimized later."""
    data_dir = Path(data_dir)

    dataset_path = list(data_dir.glob("*"))  # dataset path
    dataset_num = len(dataset_path)  # dataset number
    logger.info(f"{dataset_num} datasets detected.")

    pbar = tqdm(range(0, dataset_num, chunk_size))
    for i in pbar:
        pbar.set_description(f"Processing Chunk {i // chunk_size}")
        process_chunk = dataset_path[i:i+chunk_size]
        merged_chunk_dataset = concatenate_datasets(
            Parallel(n_jobs=nproc)(
                delayed(load_from_disk)(str(ds)) for ds in tqdm(process_chunk)
            )
        )
        merged_chunk_dataset.save_to_disk(str(data_dir / "tmp" / f"chunk_{i}"), num_proc=nproc, max_shard_size=max_shard_size)

    tmp_fp = list((data_dir / "tmp").glob("*"))
    merged_datasets = concatenate_datasets(
        Parallel(n_jobs=nproc)(
            delayed(load_from_disk)(str(ds)) for ds in tqdm(tmp_fp, desc="Final Merging")
        )
    )
    merged_datasets.save_to_disk(save_dir, num_proc=nproc, max_shard_size=max_shard_size)
    shutil.rmtree(data_dir / "tmp")


###################
##### WangLab #####
###################

class WangLabTrainDataPathExtractor(TrainDataPathExtractor):
    def get_raw_counts_fp(self, h5ad_dir: Union[Path, str]):
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

        self.train_data_path["data"].extend(list(set(fr_path_list) - set(fn_path_list)))

    def count_train_cells_from_h5ad(self, h5ad_fp: Path) -> int:
        return sc.read_h5ad(h5ad_fp, backed="r").raw.shape[0]

    def __call__(
        self, 
        h5ad_dirs: Union[List[str], str], 
        save_dir: Optional[Union[Path, str]] = None,
        count_total_train_cells: bool = False
    ):
        r"""
        **Usage:**
        >>> from stella.data import WangLabTrainDataPathExtractor
        >>> data = [
        ...     "/fse/DC/human/qc_final_tmp",          # round 1
        ...     "/fse/DC/human/second_qc_final",       # round 2
        ... ]
        >>> extractor = WangLabTrainDataPathExtractor(nproc=32)
        >>> extractor(data, save_dir="./stella", count_total_train_cells=True)
        """
        if isinstance(h5ad_dirs, str):
            h5ad_dirs = [h5ad_dirs]
        
        all_h5ad = []
        for h5ad_dir in h5ad_dirs:
            self.get_raw_counts_fp(h5ad_dir)
            all_h5ad.extend(self.find_h5ad(h5ad_dir))
        
        self.train_data_path["data"] = list(set(self.train_data_path["data"]) & set(all_h5ad))  # double check
        logger.info(f"Total training samples: {len(self.train_data_path['data'])}")
        
        if count_total_train_cells:
            counts = Parallel(n_jobs=self.nproc)(delayed(self.count_train_cells_from_h5ad)(f) \
                                                 for f in tqdm(self.train_data_path["data"]))
            self.train_data_path["ncells"] = sum(counts)
            logger.info(f"Total training cells: {self.train_data_path['ncells']:,d}")

        # save
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(save_dir) / "train_data_path.pkl", "wb") as f:
                pickle.dump(self.train_data_path, f)


if __name__ == "__main__":
    pass