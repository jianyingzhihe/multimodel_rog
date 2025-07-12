python ./src/predict/predict_answer.py \
    --data_dir data/OKVQA \
    --split train\
    --beam_size 3 \
    --model_name llama \
    --predict_path results/multimodal \
    --rule_path results/gen_rule_path/OKVQA/llama/train/predictions_3_False_train.jsonl\
    --model_path multimodels/meta-llama/llama \
    --engine_type vllm