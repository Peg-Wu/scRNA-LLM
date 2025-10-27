import sys
sys.path.append("../../src")

import evaluate
import numpy as np
from datasets import load_from_disk
from stella import STELLADataCollatorV1
from dataclasses import dataclass, field
from stella.models import STELLAForSequenceClassification
from stella.tokenizer import TranscriptomeTokenizerForCellClassification
from transformers import set_seed, Trainer, TrainingArguments, HfArgumentParser


@dataclass
class DataArguments:
    has_tokenized: bool = False
    h5ad_fp: str = None
    hf_dataset: str = None
    nproc: int = field(
        default=8,
        metadata={
            "help": "Number of processes for tokenizing h5ad file."
        }
    )

    def __post_init__(self):
        if not self.has_tokenized and self.h5ad_fp is None:
            raise ValueError("Please specify your h5ad file if you did not tokenize your data before.")
        if self.has_tokenized and self.hf_dataset is None:
            raise ValueError("Please specify your tokenized dataset.")
        if not self.has_tokenized and self.hf_dataset is None:
            raise ValueError("Although you did not tokenize your data before, you should still specify `hf_dataset` for saving your tokenized data.")


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = None
    freeze_first_n_layers: int = field(
        default=0,
        metadata={
            "help": "freeze first n layers while fine-tune, `0` means no freeze."
        }
    )


def main():
    # init args
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()
    data_args: DataArguments
    model_args: ModelArguments
    train_args: TrainingArguments
    
    # set seed
    set_seed(train_args.seed)


    # tokenize your data
    if not data_args.has_tokenized:
        tokenizer = TranscriptomeTokenizerForCellClassification(
            seed=train_args.seed,
            nproc=data_args.nproc,
            max_length=2048,
            custom_attr_name_dict={"celltype": "celltype"}
        )
        
        ds = tokenizer(
            h5ad_file=data_args.h5ad_fp,  # your h5ad file path
            save_dir=data_args.hf_dataset  # save your tokenized dataset
        )
    else:
        ds = load_from_disk(data_args.hf_dataset)


    # Process celltype labels
    uniq_ct = np.unique(ds["celltype"]).tolist()  # unique celltype
    ct2label = dict(zip(uniq_ct, range(len(uniq_ct))))  # celltype: label_id

    def process_func(example):
        example["labels"] = ct2label[example["celltype"]]
        return example

    with train_args.main_process_first(desc="Add CellType Labels"):
        ds = ds.map(process_func, num_proc=8, remove_columns=["celltype"])
        
    ds = ds.class_encode_column("labels")


    # Split the dataset into a training set, a validation set, and a test set
    ds = ds.shuffle(seed=train_args.seed)

    train_test_split = ds.train_test_split(test_size=0.2, seed=train_args.seed, stratify_by_column="labels")
    train_ds, test_ds = train_test_split["train"], train_test_split["test"]

    train_validation_split = train_ds.train_test_split(test_size=0.1, seed=train_args.seed, stratify_by_column="labels")
    train_ds, validation_ds = train_validation_split["train"], train_validation_split["test"]

    # train_size, validation_size, test_size
    if train_args.local_rank == 0:
        print(f"train size: {train_ds.shape[0]}")
        print(f"valid size: {validation_ds.shape[0]}")
        print(f"test size: {test_ds.shape[0]}")


    # Load Pretrained Model
    model = STELLAForSequenceClassification.from_pretrained(
        model_args.pretrained_model_name_or_path, 
        num_labels=len(ct2label)
    )


    # If you don't have enough GPU memory, try freezing some layers
    def freeze_first_k_layers(k=4):
        for name, param in model.named_parameters():
            if any(f"stella.encoder.layer.{i}" in name for i in range(k)):
                param.requires_grad = False

    # freeze the first k layers
    freeze_first_k_layers(k=model_args.freeze_first_n_layers)  # no freeze by default

    # check the trainable status of the parameters
    if train_args.local_rank == 0:
        for name, params in model.named_parameters():
            print(name, "\t", params.requires_grad)


    # Start Training
    clf_metrics = evaluate.combine(
        [
            "../../src/stella/metrics/accuracy",
            "../../src/stella/metrics/precision",
            "../../src/stella/metrics/recall",
            "../../src/stella/metrics/f1",
        ]
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return clf_metrics.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=STELLADataCollatorV1,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Performance on validation dataset
    eval_res = trainer.evaluate(validation_ds)
    if train_args.local_rank == 0:
        print("="*50)
        print(eval_res)
        print("="*50)

    # Performance on test dataset
    test_res = trainer.predict(test_ds)
    if train_args.local_rank == 0:
        print("="*50)
        print(test_res)
        print("="*50)


if __name__ == "__main__":
    main()