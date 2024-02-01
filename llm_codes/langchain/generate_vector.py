from transformers import BertTokenizer, BertModel
import torch
import json
import faiss
import numpy as np
import pickle
def normal(vector):
    ss=sum([s**2 for s in vector])
    return [round(s/ss,5) for s in vector]
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    result= torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    result=result[0].tolist()
    result=normal(result)
    return result
def get_vector(sentence):
    encoded_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

model_name="text2vec"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
if __name__ == "__main__":
# Load model from HuggingFace Hub
    id_knowadge={}
    with open("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2/data/wenlv_token.jsonl",encoding="utf-8") as f:
        lines=[json.loads(s.strip()) for s in f.readlines()]
    id_vector=[]
    for i,data in enumerate(lines):
        query=data["context"].replace("Instruction: ","").replace("\nAnswer: ","")
        target=data["target"]
        #使用bert模型把问题进行向量化
        vector=get_vector(query)
        #每条知识，对应一个id
        #id和向量
        id_vector.append(vector)
        #id和对应的知识
        id_knowadge[i]={"query":query,"target":target}
    id_vector=np.array(id_vector)
    with open("id_knowadge","w") as f:
        json.dump(id_knowadge,f,ensure_ascii=False)
    #id和向量用faiss库来存储
    #faiss是向量数据库，支持向量的高性能检索
    index = faiss.IndexFlatL2(768) 
    # print(index.is_trained)
    index.add(id_vector)    
    # print(index.ntotal)
    with open("id_vector","wb") as f:
        pickle.dump(index,f)