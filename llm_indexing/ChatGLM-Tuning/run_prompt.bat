python finetune_prompt.py ^
    --do_train ^
    --train_file E:\code\chatglm\ChatGLM-Tuning\data\wenlv_token ^
    --prompt_column content ^
    --response_column summary ^
    --overwrite_cache ^
    --model_name_or_path E:\code\chatglm\chatglm2 ^
    --output_dir chatglm-6b-prompt ^
    --overwrite_output_dir ^
    --per_device_train_batch_size 16 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 1 ^
    --predict_with_generate ^
    --max_steps 200 ^
    --learning_rate 2e-2 ^
    --quantization_bit 4 ^
    --pre_seq_len 128 ^
    --remove_unused_columns False

