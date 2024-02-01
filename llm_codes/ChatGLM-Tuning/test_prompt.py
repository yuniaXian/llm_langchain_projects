
from transformers import AutoTokenizer, AutoModel
#分词器 仍然用原生的
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
#加载训练好的模型模型
#打上外挂的模型
model1 = AutoModel.from_pretrained("chatglm-6b-prompt-bak", trust_remote_code=True).cuda()
print ("prefix-tuning",model1)
#model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()
#原生模型

model2 = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()
print ("原生",model2)
# input="组织大学部门六七个人去武功山旅游，求攻略，路线"
# response, history = model.chat(tokenizer, input, history=[])
# print(response)

