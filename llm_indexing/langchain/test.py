import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import math
import torch
from transformers import AutoTokenizer, AutoModel,AutoConfig
import numpy as np
from torch import nn
import json
import re
def cal_sentence_info(model, tokenizer, config,sentence):

    input_ids = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True,padding=True,add_special_tokens=True)
    target_ids = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True,padding=True,add_special_tokens=False)['input_ids'][0].tolist()+[config.eos_token_id]
    #输入长度
    input_length=len(input_ids['input_ids'][0].tolist())
    #目标长度
    target_length=len(target_ids)
    start=input_length-target_length
    # output=model.cal_logits(**input_ids,return_dict=True,output_attentions=False,output_hidden_states=False)[0]
    output=model.cal_logits(**input_ids)[0]
    results=[]
    for i in range(0,target_length):
        target=target_ids[i]
        p=nn.functional.softmax(output[start+i])[target]
        results.append(-1*math.log(p))
    score=np.mean(results)
    return score
def cut_sentences(content):
	# 结束符号，包含中文和英文的
	end_flag = ['?', '!', '.', '？', '！', '。', '…']
	
	content_len = len(content)
	sentences = []
	tmp_char = ''
	for idx, char in enumerate(content):
		# 拼接字符
		tmp_char += char

		# 判断是否已经到了最后一位
		if (idx + 1) == content_len:
			sentences.append(tmp_char)
			break
			
		# 判断此字符是否为结束符号
		if char in end_flag:
			# 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
			next_idx = idx + 1
			if not content[next_idx] in end_flag:
				sentences.append(tmp_char)
				tmp_char = ''		
	return sentences
def compress_sentence(text,model,tokenizer,config,max_length=200):
    sentences=cut_sentences(text)
    sentences=[s[0:max_length] for s in sentences]
    results=[]
    for i,sentence in enumerate(sentences):
        p=cal_sentence_info(model, tokenizer,config,sentence)
        results.append([sentence,i,p])
    results=sorted(results,key=lambda s:s[2],reverse=True)
    results2=[]
    l=0
    for sentence,i,_ in results:
        l+=len(sentence)
        if l>max_length:
              break
        else:
              results2.append([sentence,i])
    results2=sorted(results2,key=lambda s:s[1])
    results2=[s[0] for s in results2]
    return "".join(results2)
#分词器 仍然用原生的
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True)
    #model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).float()
    config = AutoConfig.from_pretrained( "E:\code\chatglm\chatglm2", trust_remote_code=True, device_map='auto')
    sentence="中国的首都是北京"
    #给模型的输入(问题是一个孔字符串)
    input_ids = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True,padding=True,add_special_tokens=True)
    #希望模型预测出来的结果
    target_ids = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True,padding=True,add_special_tokens=False)['input_ids'][0].tolist()+[config.eos_token_id]
    print("input",input_ids)
    print ("target_ids",target_ids)

     


