from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# https://www.tizi365.com/topic/10092.html
label = ""
model_local_path ='/Users/jxian3/axp/rim/novanlp/dev/jxian3/models/bge-large-en'
data_path = f"/Users/jxian3/axp/rim/novanlp/dev/jxian3/projects/nova-nlp/tasks/kg_intent/{label}.csv" 
# sentences clustering
df = pd.read_csv(data_path)
lst_text = df.tolist()
sentences_1 = []
sentences_2 = []
model = SentenceTransformer(model_local_path)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# s2p (short query to long passage)

queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T

# Using langchain
from langchain.embeddings import HuggingFaceBgeEmbeddings
# model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_local_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："

# Using HuggingFace Transformers
# the last hidden state of the first token [CLS] is chosen as the sentence embedding

from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["What is my statement balance?", "How much do I owe you?"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_local_path)
model = AutoModel.from_pretrained(model_local_path)
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)