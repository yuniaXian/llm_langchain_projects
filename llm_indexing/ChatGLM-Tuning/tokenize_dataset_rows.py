import argparse
import json
from tqdm import tqdm

import datasets
import transformers
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments,TrainerCallback
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

def preprocess(tokenizer, config, example, max_seq_length):
    #问题
    prompt = example["context"]
    #答案
    target = example["target"]
    #问题分词
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    #答案分词
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    #input_ids:问题分词+答案分词  seq_len:答案长度
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

model_name = "E:\code\chatglm\chatglm2"
tokenizer = transformers.AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2",trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained( model_name, trust_remote_code=True, device_map='auto')

def read_jsonl(path, max_seq_length, skip_overlength=False):
    # print ("sss")
    # model_name = "E:\code\chatglm\chatglm2"
    # tokenizer = transformers.AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2",trust_remote_code=False)
    # print ("bbb")
    # config = transformers.AutoConfig.from_pretrained( model_name, trust_remote_code=False, device_map='auto')
    # print ("aaa")
    # asd
    with open(path, "r",encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
