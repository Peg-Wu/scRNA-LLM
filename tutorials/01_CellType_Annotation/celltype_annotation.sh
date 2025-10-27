output_dir="./cta_running_log"
h5ad_fp="./zheng68k/zheng68k.h5ad"
hf_dataset="./tokenized_zheng68k_b100"
pretrained_model_name_or_path="../../pretrained_models/B100_L2048"
deepspeed_config="./ds_zero2.json"


DATAPARAMS="
    --has_tokenized=True \
    --h5ad_fp=$h5ad_fp \
    --hf_dataset=$hf_dataset \
    --nproc=8"


MODELPARAMS="
    --pretrained_model_name_or_path=$pretrained_model_name_or_path \
    --freeze_first_n_layers=0"


TRAINPARAMS="
    --seed=42 \
    --output_dir=$output_dir \
    --overwrite_output_dir=True \
    --report_to="tensorboard" \
    --logging_steps=10 \
    --eval_strategy="epoch" \
    --save_strategy="epoch" \
    --load_best_model_at_end=True \
    --metric_for_best_model="f1" \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=64 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --max_grad_norm=1.0 \
    --optim="adamw_torch" \
    --bf16=True \
    --dataloader_pin_memory=True \
    --dataloader_num_workers=8 \
    --dataloader_persistent_workers=True \
    --ddp_find_unused_parameters=True \
    --gradient_checkpointing=False"


RADOM_PORT=$(shuf -i 1024-65535 -n 1)
echo "使用的端口号: $RADOM_PORT"


# deepspeed
deepspeed --include=localhost:2,3 --master_port=$RADOM_PORT 01_CellType_Annotation.py \
    --deepspeed=$deepspeed_config \
    $MODELPARAMS \
    $DATAPARAMS \
    $TRAINPARAMS