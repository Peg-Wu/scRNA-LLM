# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import sys
sys.path.append("../")

import os
import logging

from pathlib import Path
from transformers import ( 
    Trainer,
    TrainingArguments, 
    HfArgumentParser, 
    set_seed
)
from datasets import load_from_disk
from stella import STELLADataCollatorV2
from stella.utils import print_trainable_parameters
from stella.arguments import DataArguments, ModelArguments
from stella.models import STELLAConfig, STELLAForMaskedLM

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16 or training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    config = STELLAConfig(
        hidden_size=model_args.hidden_size,
        num_attention_heads=model_args.num_attention_heads,
        num_hidden_layers=model_args.num_hidden_layers,
        intermediate_size=model_args.intermediate_size,
        moe_intermediate_size=model_args.moe_intermediate_size,
        first_k_dense_replace=model_args.first_k_dense_replace,
        moe_layer_freq=model_args.moe_layer_freq,
        hidden_act=model_args.hidden_act,
        seq_aux=model_args.seq_aux,
        aux_loss_alpha=model_args.aux_loss_alpha,
        num_experts_per_tok=model_args.num_experts_per_tok,
        n_routed_experts=model_args.n_routed_experts,
        n_shared_experts=model_args.n_shared_experts,
        scoring_func=model_args.scoring_func,
        norm_topk_prob=model_args.norm_topk_prob,
        attention_dropout=model_args.attention_dropout,
        initializer_range=model_args.initializer_range,
        rms_norm_eps=model_args.rms_norm_eps,
        pretraining_tp=model_args.pretraining_tp
    )

    logger.info('Config: %s', config)

    model = STELLAForMaskedLM(config)

    if training_args.local_rank == 0:
        print("="*100)
        print_trainable_parameters(model)
        print("="*100)

    # TODO: Load Train Dataset
    train_dataset = load_from_disk(data_args.data_path)
    train_dataset = train_dataset.shuffle(training_args.seed)

    logger.info(f"Load train dataset success! Total training cells ({len(train_dataset):,d})!")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=STELLADataCollatorV2(mlm=True, mlm_probability=data_args.mlm_probability)
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(os.path.join(training_args.output_dir, "models"))
    trainer.save_state()


if __name__ == "__main__":
    main()