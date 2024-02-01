
from transformers import AutoTokenizer, AutoModel
#分词器 仍然用原生的
tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
#加载训练好的模型模型
model = AutoModel.from_pretrained("chatglm-6b-freeze", trust_remote_code=True).cuda()
#model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()

input="组织大学部门六七个人去武功山旅游，求攻略，路线"
response, history = model.chat(tokenizer, input, history=[],max_length=50)
print(response)

