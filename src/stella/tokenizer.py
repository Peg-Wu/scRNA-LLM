import random
import pickle
import logging
import numpy as np
import pandas as pd
import scanpy as sc

from enum import Enum
from pathlib import Path
from tqdm.auto import tqdm
from anndata import AnnData
from datasets import Dataset
from collections import Counter
from typing import Union, Optional


VOCAB = Path(__file__).parent / "gene2id.pkl"
PRETRAIN_DATA_PATH = Path(__file__).parent / "train_data_path.pkl"
BIN_BOUNDARY = Path(__file__).parent / "bin_100.pkl"


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class TranscriptomeTokenizer:
    def __init__(
        self,
        seed: int = 42,
        nproc: int = 1,
        max_length: int = 4096,
        do_qc: bool = False,
        do_normalize: bool = True,
        custom_attr_name_dict: Optional[dict] = None,
        token_dictionary_file: Union[Path, str] = VOCAB,
        train_data_path_file: Union[Path, str] = PRETRAIN_DATA_PATH,
        bin_boundary_file: Union[Path, str] = BIN_BOUNDARY
    ):
        r"""
        **Parameters:**
        seed : int
            | Random seed.
        nproc : int
            | Number of processes to use for dataset mapping.
        max_length : int
            | Max input size of model to truncate input to.
        do_qc : bool
            | Whether to conduct qc. (The qc function in this class is used by default)
            | Outcome: some cells and genes will be removed.
            | If you want to change the qc standard, inherit the `qc` function and modify it.
        do_normalize : bool
            | Whether to conduct `sc.pp.normalize_total` and `sc.pp.log1p`. (The normalize function in this class is used by default)
            | This step is the operation before dividing the expression value into bins.
        custom_attr_name_dict : Optional[dict]
            | Dictionary of custom attributes to be added to the dataset.
            | Keys are the names of the attributes in the h5ad file.
            | Values are the names of the attributes in the dataset.
        token_dictionary_file : Union[Path, str]
            | Path to pickle file containing token dictionary. (Gene Symbol: token)
        train_data_path_file : Union[Path, str]
            | Path to pickle file containing training data path. (Generate by `stella.data.TrainDataPathExtractor`)
        bin_boundary_file : Union[Path, str]
            | Path to pickle file containing bin boundary. (Generate by `stella.data.GlobalBinExpression`)
        """
        # Fix random seed
        random.seed(seed)

        self.nproc = nproc
        self.max_length = max_length
        self.do_qc = do_qc
        self.do_normalize = do_normalize
        self.custom_attr_name_dict = custom_attr_name_dict

        # load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.gene2id = pickle.load(f)
        
        # load train data path
        with open(train_data_path_file, "rb") as f:
            self.train_data_path = pickle.load(f)["data"]

        # logging
        if len(self.train_data_path) == 0:
            logging.error("No files need to be tokenized! Check your train_data_path_file!")
            raise
        else:
            logging.info(f"{len(self.train_data_path)} files need to be tokenized!")
        
        # load bin boundary
        with open(bin_boundary_file, "rb") as f:
            self.bin_boundary = pickle.load(f)
        self.nbins = len(self.bin_boundary) - 1


    def read_h5ad_raw_counts(self, h5ad_file: Path) -> sc.AnnData:
        adata = sc.read_h5ad(h5ad_file)
        return adata


    def tokenize_anndata(
        self,
        h5ad_file: Path,
        split_into_bins: bool = True
    ) -> Dataset:
        """ QC -> Select Genes in Vocab -> Normalize & log1p -> Bin -> Fix Length """

        # Load adata
        adata = self.read_h5ad_raw_counts(h5ad_file)

        # QC
        if self.do_qc:
            self.qc(adata)

        # Select Genes in Vocab
        genes_in_vocab = self.gene2id.keys()
        adata = adata[:, adata.var_names.isin(genes_in_vocab)].copy()

        # Normalize & log1p
        if self.do_normalize:
            self.normalize_and_log1p(adata)
        
        # Bin (only nonzero expression will be operated!)
        if split_into_bins:
            self.bin(adata)

        # Add Cell Meta Information
        if self.custom_attr_name_dict is not None:
            # Init file_cell_metadata
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.values()
            }
            # Fill-in file_cell_metadata
            for k in file_cell_metadata.keys():
                v = {j: i for i, j in self.custom_attr_name_dict.items()}[k]
                file_cell_metadata[k] += adata.obs[v].tolist()
        else:
            file_cell_metadata = None
        

        # Extract non-zero gene expression and their corresponding gene symbols
        def split_array_custom_sizes(arr, sizes):
            indices = np.cumsum(sizes)[:-1]
            return np.split(arr, indices)
        
        row, col = adata.X.nonzero()
        counter = Counter(row.tolist())
        sizes = [i[-1] for i in sorted(list(counter.items()), key=lambda x: x[0], reverse=False)]
        nonzero_col_indices = split_array_custom_sizes(col, sizes)
        nonzero_col_indices = list(map(lambda x: x.tolist(), nonzero_col_indices))
        for i, each in enumerate(nonzero_col_indices):
            each.insert(0, i)
        
        init_dict = {
            "nonzero_col_indices": nonzero_col_indices  # [[row_idx, nonzero_col_idx], ...]
        }

        tmp_data = init_dict | file_cell_metadata if file_cell_metadata is not None else init_dict
        
        ds = Dataset.from_dict(tmp_data)

        def format_cell_features(example):
            row_idx = example["nonzero_col_indices"][0]
            col_idx = example["nonzero_col_indices"][1:]
            random.shuffle(col_idx)  # inplace operation

            input_ids_gene_symbol = adata.var_names[col_idx][:self.max_length].map(self.gene2id).tolist()
            input_ids_gene_expression = adata.X[row_idx, col_idx].toarray()[0][:self.max_length]
            length = len(input_ids_gene_symbol)

            example["input_ids_gene_symbol"] = input_ids_gene_symbol
            example["input_ids_gene_expression"] = input_ids_gene_expression
            example["length"] = length
            return example

        ds = ds.map(format_cell_features, num_proc=self.nproc, remove_columns="nonzero_col_indices")
        return ds


    def __call__(self, save_dir: Union[Path, str] = "./", split_into_bins: bool = True):
        """
        **Examples:**
            >>> from stella.tokenizer import TranscriptomeTokenizer
            >>> tokenizer = TranscriptomeTokenizer(nproc=8, train_data_path_file=...)
            >>> tokenizer(save_dir=...)
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(tqdm(self.train_data_path)):
            ds = self.tokenize_anndata(f, split_into_bins)
            save_path = str(Path(save_dir) / str(i))
            ds.save_to_disk(save_path, num_proc=self.nproc)


    def bin(
        self,
        adata: sc.AnnData
    ):
        # TODO: 此处需要进行更改, 需要对最小值和最大值进行cutoff, 并且需要将最大bin的右区间改成闭区间!
        # 最简单的处理方法:
        # 先用np.digitize分bin
        # 如果有expression被分成第0个bin, 则将其变成第1个bin
        # 如果有expression被分成第nbins+1个bin, 则将其变成第nbins个bin
        nonzero_loc_before_bin = adata.X != 0
        adata.X[adata.X != 0] = np.digitize(adata.X[adata.X != 0], self.bin_boundary, right=False)
        nonzero_loc_after_bin = adata.X != 0
        bin_to_zero_loc = np.logical_xor(nonzero_loc_before_bin.toarray(), nonzero_loc_after_bin.toarray())
        adata.X[bin_to_zero_loc] = 1
        adata.X[adata.X == self.nbins + 1] = self.nbins
        adata.X = adata.X.astype(np.int16)


    @staticmethod
    def qc(
        adata: sc.AnnData,
        nFeature: int = 500, 
        nCount: int = 1000,
        gene_frac: float = 0.01
    ):
        """ QC standards are set by `Hailin Wei`. """
        
        if nFeature == 0 and nCount == 0:
            logging.warning("No cells/genes are removed!")
            return

        sc.pp.filter_cells(adata, min_genes = nFeature)

        if adata.shape[0] != 0:
            sc.pp.filter_cells(adata, min_counts = nCount)
        else:
            raise ValueError("No cells are retained, please change the quality control standard!")
        
        if adata.shape[1] != 0:
            sc.pp.filter_genes(adata, min_cells = int(gene_frac * adata.shape[0]))
        else:
            raise ValueError("No genes are retained, please change the quality control standard!")
        
        if 0 in adata.shape:
            raise ValueError("No cells/genes are retained, please change the quality control standard!")


    @staticmethod
    def normalize_and_log1p(
        adata: sc.AnnData,
        target_sum: bool = 1e4,
        log1p: bool = True  
    ):
        sc.pp.normalize_total(adata, target_sum=target_sum)

        if log1p:
            sc.pp.log1p(adata)


class TranscriptomeTokenizerForCellClassification(TranscriptomeTokenizer):
    def __init__(
        self,
        seed: int = 42,
        nproc: int = 1,
        max_length: int = 2048,
        do_qc: bool = False,
        do_normalize: bool = True,
        custom_attr_name_dict: Optional[dict] = None,
        token_dictionary_file: Union[Path, str] = VOCAB,
        bin_boundary_file: Union[Path, str] = BIN_BOUNDARY
    ):
        # Fix random seed
        random.seed(seed)

        self.nproc = nproc
        self.max_length = max_length
        self.do_qc = do_qc
        self.do_normalize = do_normalize
        self.custom_attr_name_dict = custom_attr_name_dict

        # load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.gene2id = pickle.load(f)

        # load bin boundary
        with open(bin_boundary_file, "rb") as f:
            self.bin_boundary = pickle.load(f)
        self.nbins = len(self.bin_boundary) - 1


    def read_h5ad_raw_counts(self, h5ad_file: Union[Path, AnnData]) -> sc.AnnData:
        if not isinstance(h5ad_file, AnnData):
            adata = sc.read_h5ad(h5ad_file)
        else:
            adata = h5ad_file

        # Rearrange gene_symbol, if gene_symbol is same, only keep first.
        adata.var_names = adata.var_names.str.replace(r'\.\d+$', '', regex=True)
        duplicated_genes = adata.var_names.duplicated(keep="first")
        adata = adata[:, ~duplicated_genes].copy()

        return adata


    def __call__(
        self, 
        h5ad_file: Union[Path, AnnData], 
        save_dir: Optional[str] = None, 
        split_into_bins: bool = True
    ):
        ds = self.tokenize_anndata(h5ad_file, split_into_bins)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(save_dir)
        return ds


###################
##### WangLab #####
###################

class WangLabTranscriptomeTokenizer(TranscriptomeTokenizer):
    def read_h5ad_raw_counts(self, h5ad_file: Path) -> sc.AnnData:
        adata = sc.read_h5ad(h5ad_file).raw.to_adata()  # WangLab raw counts data location!
        return adata


class SELECT_GENES_MODE(Enum):
    USE_HVG = "hvg"
    SPECIFY_GENE_LIST = "specify"
    USE_HVG_AND_SPECIFY_GENE_LIST = "hvg2specify"
    ALL = "all"


class Preprocessor(TranscriptomeTokenizerForCellClassification):
    def __init__(
        self,
        filter_gene_by_counts: Union[int, bool] = False,  # False or int
        filter_cell_by_counts: Union[int, bool] = False,  # False or int
        select_genes_mode: str = "hvg",
        nhvgs: Optional[int] = None,
        hvg_flavor: str = "seurat",
        gene_list: Optional[list[str]] = None,
        token_dictionary_file: Union[Path, str] = VOCAB,
        bin_boundary_file: Union[Path, str] = BIN_BOUNDARY
    ):
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.select_genes_mode = select_genes_mode
        self.nhvgs = nhvgs
        self.hvg_flavor = hvg_flavor
        self.gene_list = gene_list

        # load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.gene2id = pickle.load(f)

        # load bin boundary
        with open(bin_boundary_file, "rb") as f:
            self.bin_boundary = pickle.load(f)
        self.nbins = len(self.bin_boundary) - 1

        # check your attributes
        self.check_attr()

    
    def check_attr(self):
        if self.select_genes_mode == SELECT_GENES_MODE.USE_HVG.value and self.nhvgs is None:
            raise ValueError("Please specify the number of highly variable genes (nhvgs).")
        
        if self.select_genes_mode == SELECT_GENES_MODE.SPECIFY_GENE_LIST.value and self.gene_list is None:
            raise ValueError("Please specify the gene list you have selected (gene_list).")
        
        if self.select_genes_mode == SELECT_GENES_MODE.USE_HVG_AND_SPECIFY_GENE_LIST.value:
            assert self.nhvgs is not None and self.gene_list is not None, \
                "Please specify the number of highly variable genes (nhvgs) and the gene list you have selected (gene_list)."
    

    @staticmethod
    def _intersection_two_gene_lists(
        adata_gene_list: list, 
        specify_gene_list: list
    ):
        intersection = list(set(adata_gene_list) & set(specify_gene_list))
        if len(intersection) == 0:
            raise ValueError("None of the genes in the specified gene_list are present in the processed adata.")
        elif len(intersection) == len(specify_gene_list):
            print("All genes in the specified gene_list are present in the processed adata.")
        else:
            print(f"Among the genes in the specified gene_list, "
                  f"{len(intersection)} are present in the processed adata, "
                  f"while {len(specify_gene_list) - len(intersection)} are not.")
        return intersection
    

    def __call__(self, h5ad_file: Union[Path, AnnData], split_into_bins: bool = True):
        # Read adata
        adata = self.read_h5ad_raw_counts(h5ad_file)

        # QC
        ## step 1: filter genes
        None if isinstance(self.filter_gene_by_counts, bool) else sc.pp.filter_genes(
            adata, min_counts=self.filter_gene_by_counts
        )

        ## step 2: filter cells
        None if isinstance(self.filter_cell_by_counts, bool) else sc.pp.filter_cells(
            adata, min_counts=self.filter_cell_by_counts
        )

        # Select Genes in Vocab
        genes_in_vocab = self.gene2id.keys()
        adata = adata[:, adata.var_names.isin(genes_in_vocab)].copy()

        # Normalize & log1p
        self.normalize_and_log1p(adata)
            
        # Fix genes according to `self.select_genes_mode`
        if self.select_genes_mode == SELECT_GENES_MODE.USE_HVG.value:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.nhvgs, flavor=self.hvg_flavor)
            adata = adata[:, adata.var["highly_variable"].values].copy()
        elif self.select_genes_mode == SELECT_GENES_MODE.SPECIFY_GENE_LIST.value:
            filtered_gene_list = self._intersection_two_gene_lists(adata.var_names, self.gene_list)
            adata = adata[:, filtered_gene_list].copy()
        elif self.select_genes_mode == SELECT_GENES_MODE.USE_HVG_AND_SPECIFY_GENE_LIST.value:
            filtered_gene_list = self._intersection_two_gene_lists(adata.var_names, self.gene_list)
            sc.pp.highly_variable_genes(adata, n_top_genes=self.nhvgs, flavor=self.hvg_flavor)
            all_selected_genes = adata.var_names[adata.var["highly_variable"]].tolist()
            all_selected_genes.extend(filtered_gene_list)
            adata = adata[:, list(set(all_selected_genes))].copy()
        elif self.select_genes_mode == SELECT_GENES_MODE.ALL.value:
            pass  # Do nothing, keep all genes in h5ad.
        else:
            raise ValueError("Check your `select_genes_mode` attribute!")
        
        # Bin
        if split_into_bins:
            self.bin(adata)

        return adata
    

    def get_hf_dataset_from_adata(self, adata):
        input_ids_gene_symbol = adata.var_names.map(self.gene2id).tolist()
        input_ids_gene_symbol = [input_ids_gene_symbol for _ in range(len(adata))]

        init_dict = {
            "input_ids_gene_symbol": input_ids_gene_symbol,
            "input_ids_gene_expression": adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        }
        return Dataset.from_dict(init_dict)
    

if __name__ == "__main__":
    pass