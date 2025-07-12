#!/bin/bash

swift sft \
  --model multimodels/meta-llama/llama \
  --train_type lora \
  --dataset src/finetuning/converted_output_llama.json \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --max_length 2048 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --output_dir ./output-lora \
  --save_strategy epoch \
  --logging_steps 10 \
  --bf16 True \
