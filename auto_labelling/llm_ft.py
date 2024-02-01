import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_for_int8_training
from transformers import AutoModelForCasualLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# >>> CUDA_VISIBLE_DEVICES=0 python train.py
# >>> CUDA_VISIBLE_DEVICES=0 autotrain llm --train --project_name output --model Salesforce/xgen-7b-8k-base --data_path tatsu-lab/alpaca --use_peft --use_int4 --trainer sft --learning_rate 2e-4

def train():
    train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # input,
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCasualLM.from_pretrained("Salesforce/xgen-7b-8k-base", load_in_4big=True, torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_for_int8_training(model) 

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CASUAL_LM")
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments (
        output_dir="xgen-7b-tuned-alpaca",
        per_device_train_batch_size=4,
        optim="adamw_torch",
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=True,
        )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config
        )
    
