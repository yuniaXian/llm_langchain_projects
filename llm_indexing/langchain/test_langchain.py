
from transformers import AutoTokenizer, AutoModel
import numpy as np
import generate_vector
import pickle
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def get_prompt(sentences,query):
    """
    Design a template to re-struct the samples
    """
    # Example: return "Please answer the following questions based on {}。the questions is ：{}".format("。".join(sentences),query)
    return "请根据以下事实回答问题{}。问题是：{}".format("。".join(sentences),query)

def get_prompt_label_defs(label_defs, labels):

    formated = ""
    for label_def, label in zip(label_defs, labels):
        formatted += f"{label_def} is labelled as {label}. "
    return formatted + " Assign label to the following samples based on instructions above: \n"

def get_prompt_labelling(label_defs, labels, samples, Max_length):

    #TODO trucate the return based on Max_length
    instructions = get_prompt_label_defs(label_defs, labels)
    res = [f"{idx}. "+ sample for idx, sample in enumerate(samples)]
    return instructions + "\n".join(res)


# tokenizer = AutoTokenizer.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True)
# model = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True).cuda()
# data['142'] = data['142'].decode('latin-1')
with open("llm_codes/langchain/id_vector","rb") as f:
    index=pickle.load(f)
with open("llm_codes/langchain/id_know", "r", encoding= 'unicode_escape') as f:
    id_know=json.load(f)
input="纽约有什么好玩的地方"
# query to vector
vector=generate_vector.get_vector(input) # get sentence_embeddings
vector=np.array([vector])
# search the closed 3 ids from faiss library. 找到最近的3个id  D距离，I id
D, I = index.search(vector, 3)
D=D[0]
I=I[0]
sentences=[]
for d,i in zip(D,I):
    # filter for distance
    if d>0.02:
        continue
    #print (id_know[str(i)]['query'])
    sentences.append(id_know[str(i)]['target'])
#print (sentences)
prompt=get_prompt(sentences,input)
print (prompt)
# response, history = model.chat(tokenizer, prompt, history=[])
# print(response)

