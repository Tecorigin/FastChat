export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:1024
export OMP_NUM_THREADS=4
export TORCH_SDAA_LINEAR_HIGHPREC=1
export TORCH_SDAA_BADDBMM_HIGHPREC=1
export TORCH_SDAA_BMM_HIGHPREC=1
export TORCH_SDAA_BMM_HIGHPERF=1
export TORCH_SDAA_BLAS_TRANSPOSE=0
export TORCH_SDAA_FUSED_ATTN_MEM_LIMITED=1
export TORCH_SDAA_ALIGN_NV_DEVICE=a100
export HF_ENDPOINT=https://hf-mirror.com


deepspeed fastchat/train/train_lora.py \
    --model_name_or_path /path/llama-2-7b/ \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ./data/dummy_conversation.json \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --deepspeed playground/ds_zero_3.json \
