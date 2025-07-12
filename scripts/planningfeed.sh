SPLIT="test"
DATASET_LIST="OKVQA"
MODEL_NAME=Qwen2.5-VL-7B-Instruct
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

BEAM_LIST="3" # "1 2 3 4 5"
for DATASET in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        python src/gen_rule_path_feed.py \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        -d ${DATASET} \
        --split ${SPLIT} \
        --n_beam ${N_BEAM}
    done
done
