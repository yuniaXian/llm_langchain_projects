import torch
from transformers import AutoTokenizer, AutoModel
from finetune_adapter import  Adapter,CombinedModel
#分词器 仍然用原生的
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
#加载训练好的模型模型
model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()
for i,s in enumerate(model.transformer.encoder.layers):
        adapter=torch.load("chatglm-6b-adapter/{}".format(i)).cuda()
        combined=CombinedModel(s.self_attention.dense,adapter)
        s.self_attention.dense=combined
        break

print (model)

input="江西有啥好玩的"


response, history = model.chat(tokenizer, input, history=[],max_length=200)
print(response)

