#!/bin/bash

# --- 1. Global Variables ---
BASE_MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com
MASTER_PORT_START=16990

# --- 2. Experiment Settings ---
SEEDS=(1024)
TARGET_RANK=128
TARGET_ALPHA=128

# --- 3. Initialize PiSSA weights if not exist ---
RES_MODEL="output/PiSSA-Llama-2-7b-r${TARGET_RANK}"
echo "--------------------------------------------------------------------"
echo "Verifying PiSSA initialization for Rank=${TARGET_RANK}..."
if [ -d "$RES_MODEL" ]; then
    echo "  - Found existing PiSSA initialization at: ${RES_MODEL}"
else
    echo "  - PiSSA initialization not found. Creating..."
    python utils/init_pissa.py \
        --base_model_path $BASE_MODEL \
        --output_dir $RES_MODEL \
        --init_weights pissa_niter_16 \
        --lora_r $TARGET_RANK \
        --lora_alpha $TARGET_ALPHA \
        --lora_dropout 0 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to initialize PiSSA weights. Aborting."
        exit 1
    fi
    echo "  - PiSSA weights initialized successfully for Rank=${TARGET_RANK}."
fi
echo "--------------------------------------------------------------------"

# --- 4. Main Training Loop ---
for i in "${!SEEDS[@]}"; do
    SEED_TO_SUMMON=${SEEDS[$i]}
    MASTER_PORT=$((MASTER_PORT_START + i))
    OUTPUT_PATH="output/universality_study/1ba-lora_v5_r${TARGET_RANK}_seed${SEED_TO_SUMMON}"

    echo "===================================================================="
    echo "Launching Training Run #${i+1}"
    echo "  - Seed: ${SEED_TO_SUMMON}"
    echo "  - Rank: ${TARGET_RANK}"
    echo "  - Output Path: ${OUTPUT_PATH}"
    echo "===================================================================="

    deepspeed --master_port=${MASTER_PORT} --include=localhost:0,1 train.py \
        --deepspeed configs/ds_config_zero2_no_offload.json \
        --model_name_or_path $RES_MODEL \
        --adapter_name_or_path "pissa_init" \
        --use_ba_lora True \
        --base_model_for_pt $BASE_MODEL \
        \
        --lambda1 0.025 \
        --lambda2 0.005 \
        --lambda3 0.005 \
        \
        --lambda1_schedule "cosine" \
        --lambda_focus_schedule "two_phase" \
        --lambda_warmup_ratio 0.2 \
        --lambda_ramp_up_ratio 0.05 \
        \
        --svd_k 10 \
        --top_k_entropy 20 \
        --distill_temp 2.0 \
        --svd_frob_norm True \
        --full_finetune False \
        --bf16 True \
        --seed $SEED_TO_SUMMON \
        --data_path $DATA_PATH \
        --sub_task metamath:100000 \
        --dataset_split train \
        --dataset_field instruction output \
        --output_dir ${OUTPUT_PATH} \
        --num_train_epochs 1 \
        --model_max_length 512 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --report_to "tensorboard" \
        --merge True

    # --- 5. Check Training Status ---
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for SEED: ${SEED_TO_SUMMON}. Aborting loop."
        exit 1
    fi
    
    # --- 6. Inference and Evaluation ---
    echo ""
    echo "Training complete for SEED: ${SEED_TO_SUMMON}. Generating inference results..."
    python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
    if [ $? -ne 0 ]; then echo "ERROR: Inference generation failed for SEED: ${SEED_TO_SUMMON}."; exit 1; fi

    echo ""
    echo "Inference complete. Calculating accuracy for SEED: ${SEED_TO_SUMMON}..."
    python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
    if [ $? -ne 0 ]; then echo "ERROR: Accuracy calculation failed for SEED: ${SEED_TO_SUMMON}."; exit 1; fi

done

echo "All training runs completed."

