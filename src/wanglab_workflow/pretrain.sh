data_path="/share/home/u19505/wupengpeng/STELLA/pretrain_dataset/Human_B100_L2048/merge"
output_dir="/share/home/u19505/wupengpeng/STELLA/logs/Human_B100_L2048"
deepspeed_config="./ds_zero2.json"

MODELPARAMS="
    --hidden_size=512 \
    --num_attention_heads=8 \
    --num_hidden_layers=6 \
    --intermediate_size=1024 \
    --moe_intermediate_size=1024 \
    --first_k_dense_replace=0 \
    --moe_layer_freq=1 \
    --hidden_act="silu" \
    --seq_aux=True \
    --aux_loss_alpha=0.001 \
    --num_experts_per_tok=2 \
    --n_routed_experts=8 \
    --n_shared_experts=1 \
    --scoring_func="softmax" \
    --norm_topk_prob=False \
    --attention_dropout=0.0 \
    --initializer_range=0.02 \
    --rms_norm_eps=1e-6 \
    --pretraining_tp=1"

DATAPARAMS="
    --data_path=$data_path \
    --mlm_probability=0.3"

TRAINPARAMS="
    --do_train=True \
    --do_eval=False \
    --seed=42 \
    --output_dir=$output_dir \
    --overwrite_output_dir=True \
    --report_to="tensorboard" \
    --logging_steps=200 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=64 \
    --learning_rate=1e-3 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.1 \
    --weight_decay=0 \
    --max_grad_norm=1.0 \
    --optim="adamw_torch" \
    --save_steps=0.04 \
    --bf16=True \
    --dataloader_pin_memory=True \
    --dataloader_prefetch_factor=16 \
    --dataloader_num_workers=16 \
    --dataloader_persistent_workers=True \
    --ddp_find_unused_parameters=True \
    --gradient_checkpointing=False"

RADOM_PORT=$(shuf -i 1024-65535 -n 1)
echo "使用的端口号: $RADOM_PORT"


# torchrun
# CUDA_VISIBLE_DEVICES=2,3,4,6 torchrun --nnodes=1 --nproc-per-node=4 --master_port=$RADOM_PORT w06_pretrain.py \
#     $MODELPARAMS \
#     $DATAPARAMS \
#     $TRAINPARAMS


# deepspeed
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=$RADOM_PORT w06_pretrain.py \
    --deepspeed=$deepspeed_config \
    $MODELPARAMS \
    $DATAPARAMS \
    $TRAINPARAMS