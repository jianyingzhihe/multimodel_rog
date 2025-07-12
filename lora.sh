#!/bin/bash

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 开始训练
swift sft \
  --model multimodels/meta-llama/llama \
  --train_type lora \
  --dataset swift/OK-VQA_train \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_accumulation_steps 16 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
