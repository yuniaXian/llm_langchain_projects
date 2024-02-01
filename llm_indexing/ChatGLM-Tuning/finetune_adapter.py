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
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        #问题+答案
        ids = feature["input_ids"]
        #问题长度
        seq_len = feature["seq_len"]
        #-100特殊字符，表示不预测
        # [-100] * (seq_len - 1) 问题部分是不需要预测的
        #ids[(seq_len - 1) :] 预测答案
        #[-100] * (longest - ids_l)  不零位置不需要预测
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))




class Adapter(nn.Module):
    def __init__(self, in_features, mid_features):
        super(Adapter, self).__init__() # or nn.Module.__init__(self)
        self.w1 = nn.Linear(in_features, mid_features)
        self.w2 = nn.Linear(mid_features, in_features)
        self.act=nn.ReLU()
    def forward(self, x):
        y = self.w1(x)
        y=self.act(y)
        y = self.w2(y)
        return 1e-1*y+x
#把适配器和原有的全连接层绑在一起
class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        #需要训练的，都需要使用float32
       
        self.submodel1 = submodel1.to(torch.float32)
        self.submodel2 = submodel2.to(torch.float32)
 
    def forward(self, x):
        x=x.to(torch.float32)
        y1 = self.submodel1(x)
        y2 = self.submodel2(y1)
        return y2.half()
# adapter=Adapter(in_features=10,mid_features=8)
# print (adapter)
# asd

import pickle
def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")
#  

def main():
    os.environ["WANDB_DISABLED"] = "true"   
    writer = SummaryWriter()
    training_args = TrainingArguments(output_dir="chatglm-6b-adapter",per_device_train_batch_size=10,remove_unused_columns=False,num_train_epochs=1,learning_rate=1e-5)
    dataset_path="data\wenlv_token"
    # # init model
  
    #加载训练好的模型模型
    #模型默认的是float16，进行推理没问题
    model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True,device_map="auto").cuda() 

    #冻结住模型的所有参数
    for name, param in model.named_parameters():
        param.requires_grad=False


    adapter_list={}
    for i,s in enumerate(model.transformer.encoder.layers):
        #print (s)
        m=s.self_attention.dense
        #适配器的输入向量维度
        in_features=int(m.in_features)
        #拍脑袋取
        mid_features=4
        #生成一个适配器
        adapter=Adapter(in_features=in_features,mid_features=mid_features)
        #适配器和原有的全连接层绑在一起
        #默认是放在cpu里面，所以使用cuda 放进gpu
        combined=CombinedModel(m,adapter).cuda()
        #替代原生的位置
        s.self_attention.dense=combined
        adapter_list[i]=adapter
        #因为显存不够大，只给第一层加了适配器、
        break
    

 
    # for name, param in model.named_parameters():
    #     print (name,param.requires_grad)
     
    get_trainable_para_num(model)
    print (model)

    dataset = datasets.load_from_disk(dataset_path)
    print (dataset)
    print(f"\n{len(dataset)=}\n")
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    #不能这样直接保存，直接保存不会报错
    #但是加载不起来，因为没有改配置文件
    #model.save_pretrained(training_args.output_dir)
    trainer.train()
    writer.close()
     
    for i,adapter in adapter_list.items():
        torch.save(adapter, training_args.output_dir+"/"+str(i))
    input="你好"
    response, history = model.chat(tokenizer, input, history=[],max_length=200)
    print(response,len(response))

if __name__ == "__main__":
    main()
