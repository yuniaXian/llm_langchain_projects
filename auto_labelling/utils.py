import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import os, json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def prompts_labelling(label, inst_type):
    prompts = {}
    prompts["summerization"] = [
        "The following is a set of inquiries asked by clients. Summarize what they talk about:",
        "Summarize what are the topics the above inquries are about? In summary, the above utterances above are about:",
        "Summarize what are the topics the above inquries are about? In summary, the above utterances above. ",
        "Summarize what are the topics the above inquries are about? Include clients' situation and what the want. \
            In summary, the clients ",
        "Summarize what are the topics the above inquries are about? The inquries of above utterances above are ",
    ]

    prompts["res2inst"] = [
        f"The following list of utterances are all labelled as {label}. Based on them, write a labelling instruction of label {label}. \
            So when people or chatGPT read this labelling instruction, they will label utterances with similar \
            content or similar intent as {label}:"
    ]

    prompts["intent"] = [
        "What the following utterances talking about? extract the intent or the issue they want to solve."
    ]
    # https://chat.openai.com/c/7fe8152e-4201-4880-8e15-40c491403d53

def txt2lst(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split('. ', 1)[1] if line.strip().split('. ', 1)[0].isdigit() else line.strip() for line in lines]
        return lines

def get_similarity(model, lst1, lst2):
    embeddings_1 = model.encode(lst1, normalize_embeddings=True)
    embeddings_2 = model.encode(lst2, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    return similarity

def get_max_min_col_row(data,Is_same_label=False, Is_print=True):

    N, M = data.shape[0], data.shape[1]
    if Is_same_label:
        data = data-np.eye(N,M)

    aver_row = np.sum(data, axis=0)/data.shape[0]
    aver_col = np.sum(data, axis=1)/data.shape[1]
    aver = np.sum(data)/(data.shape[0]*data.shape[1])

    
    # Calculate max, min, and average for each row
    max_per_row = np.max(data, axis=1)
    min_per_row = np.min(data, axis=1)
    avg_per_row = np.mean(data, axis=1)

    # Calculate max, min, and average for each column
    max_per_col = np.max(data, axis=0)
    min_per_col = np.min(data, axis=0)
    avg_per_col = np.mean(data, axis=0)

    # Find the index of max and min for each row
    max_idx_per_row = np.argmax(data, axis=1)
    min_idx_per_row = np.argmin(data, axis=1)

    # Find the index of max and min for each column
    max_idx_per_col = np.argmax(data, axis=0)
    min_idx_per_col = np.argmin(data, axis=0)

    # Find the second largest value in each column
    second_largest_in_columns = np.partition(data, -2, axis=0)[-2]

    # Find the second largest value in each row
    second_largest_in_rows = np.partition(data, -2, axis=1)[:, -2]
    
    if Is_same_label:
        avg_per_col = np.sum(data, axis=0)/(N-1)
        avg_per_row = np.sum(data, axis=1)/(M-1)
        aver = np.sum(aver_row)/(N-1)
        min_per_row = np.min(data + np.eye(N,M), axis=1)
        min_per_col = np.min(data + np.eye(N,M), axis=0)
        min_idx_per_row = np.argmin(data + np.eye(N,M), axis=1)
        min_idx_per_col = np.argmin(data + np.eye(N,M), axis=0)
        data = data+np.eye(N,M)
    # Print the results
    if Is_print:
        print(f"aver row: {aver_row}, aver col: {aver_col}, aver: {aver}")
        print("Max per row:", max_per_row)
        print("Min per row:", min_per_row)
        print("Average per row:", avg_per_row)

        print("Max per column:", max_per_col)
        print("Min per column:", min_per_col)
        print("Average per column:", avg_per_col)

        # Print the results
        print("Index of Max per row:", max_idx_per_row)
        print("Index of Min per row:", min_idx_per_row)

        print("Index of Max per column:", max_idx_per_col)
        print("Index of Min per column:", min_idx_per_col)

        print("Second largest value in each column:", second_largest_in_columns)
        print("Second largest value in each row:", second_largest_in_rows)

    d = {}
    d['avg_r'] = avg_per_row
    d['avg_c'] = avg_per_col
    d['min_r'] = min_idx_per_row
    d['min_c'] = min_idx_per_col
    d['max_r'] = max_idx_per_row
    d['max_c'] = max_idx_per_col
    d['avg'] = aver
    d['min'] = np.min(data)
    d['max'] = np.max(data-np.eye(N,M)) if Is_same_label else np.max(data)
    return d

def get_dict_val2idx(A):
    val2idx = defaultdict(list)
    for idx, val in enumerate(A):
        val2idx[int(val)].append(idx)
    return val2idx

def get_topk_within_idx(A, indices, topk):
    selected_values = A[indices]

    # Find the top 5 values and their corresponding indices
    top_indices = selected_values.argsort()[-topk:][::-1]  # Indices of the top k values
    top_values = selected_values[top_indices]

    return top_values, [indices[i] for i in top_indices] # top_indices 是ndarray 不能直接 indices[top_indices]

def get_topk_txt_within_idx(d_minmax, d_idx, topk=3, key='avg_r'):
    A = d_minmax[key]
    d_res = {}
    for key, idxes in d_idx.items():
        top_val, top_idxes = get_topk_within_idx(A, idxes, topk)
        d_res[key] = top_idxes
    return d_res

def visual_res(d_res, text_dict, label1):
    for key, idxes in d_res.items():
        print(key, text_dict[label1][key])
        print([[idx, text_dict[label1][idx]] for idx in idxes])

def txt_collect(d_res, extra_list, d_txt4def, label):
    idx_set = set(extra_list)
    for outliner, txt_lst in d_res.items():
        idx_set = idx_set | set(txt_lst)
        idx_set.add(outliner)
    d_txt4def[label] = sorted(list(idx_set))
    return d_txt4def

def get_txt_lst(model, label_lst, df, label_col='my_label', col='utterance'):
    """
    combine all text samples with label in the label lst
    Return: 
        combined text list: list of text samples belong to label in label list.
        mat_dict: similarity matrix of combined text list.
        text_dict: label: list of text samples.
    """
    lst = []
    text_dict = {}
    for label in label_lst:
        text_dict[label]=df[df[label_col]==label][col].tolist()
        lst += df[df[label_col]==label][col].tolist()
        print(label, len(text_dict[label]))
    mat_dict = {}
    mat_dict['combined'] = get_similarity(model, lst,lst)
    
    return mat_dict, text_dict


def get_label2st(label_lst, df, label_col='my_label', col='utterance'):
    label2st = {}
    s, t = 0, 0
    for label in label_lst:
        t = s+len(df[df[label_col]==label][col].tolist())
        label2st[label]=[s, t]
        print(f"{label}: [{s}, {t}]")
        s = t
    return label2st

def get_label2matrix(matrix, label2st, label1, label2):
    """
    matrinx: numpy ndarray of (N, N)
    """
    st1, st2 = label2st[label1], label2st[label2]
    return matrix[st1[0]:st1[1], st2[0]:st2[1]]

def get_cross_matrix_dict(label_lst, label2st, matrix):
    mat_dict = {}
    for label1 in label_lst:
        if label1 not in mat_dict:
            mat_dict[label1] = {}
            for label2 in label_lst:
                mat_dict[label1][label2] = get_label2matrix(matrix, label2st, label1, label2)
    return mat_dict

def get_txt_lst_per_label(model, label_lst, df, top_k_labels, label_col='my_label', col='utterance'):
    """
    Return:
        text_dict: label: list of text samples
        mat_dict: label1,label1: self similarity matrix within a label
    """
    mat_dict = {}
    for label in top_k_labels:
        mat_dict[label] = get_similarity(model, df[df[label_col]==label][col].tolist(),df[df[label_col]==label][col].tolist())

    text_dict = {}
    for label in df['my_label'].unique():
        text_dict[label]=df[df[label_col]==label][col].tolist()
    
    return mat_dict, text_dict

def get_crx_labels_max_min_col_row(mat_crx_dict, label_lst, Is_print=False):
    aver_rel = {label:{} for label in label_lst}
    for l1 in label_lst:
        for l2 in label_lst:
            d_minmax = get_max_min_col_row(mat_crx_dict[l1][l2], Is_same_label=(l1==l2), Is_print=False)
            aver_rel[l1][l2] = d_minmax
            print(l1, l2, d_minmax['avg'])
            if Is_print:
                print(l1," aver row: ", d_minmax['avg_r'])
                print(l1," aver col: ", d_minmax['avg_c'])
    return aver_rel


def generate_label2samples(model, df, label_lst, topk=1, Is_combined=True, path="", label_col='my_label', col='utterance'): # text_dict, label2st, mat_dict_combined, mat_crx_dict,
    if Is_combined: 
        label1, label2 = 'combined', 'combined'
        print(f"label1: {label1}, label2: {label2}")
        mat_dict_combined, text_dict=get_txt_lst(model, label_lst, df)
    elif Is_combined==False and (len(label_lst) == 2):
        label1 = label_lst[0]
        label2 = label_lst[1] #'statement_balance'
        print(f"label1: {label1}, label2: {label2}")

    elif len(label_lst) > 2:
        aver_rel = get_crx_labels_max_min_col_row(label_lst, Is_print=False)
        return aver_rel
    
    label2st= get_label2st(label_lst, df, label_col='my_label', col='utterance')
    mat_dict_combined, text_dict=get_txt_lst(model, label_lst, df)
    mat_crx_dict = get_cross_matrix_dict(label_lst, label2st, mat_dict_combined['combined'])

    if Is_combined is False:
        d_minmax = get_max_min_col_row(mat_crx_dict[label1][label2], Is_same_label=(label1==label2), Is_print=True)
    else:
        d_minmax = get_max_min_col_row(mat_dict_combined['combined'], Is_same_label=(label1==label2), Is_print=True)
        # Get outliners
        cnt = Counter(d_minmax['min_r'])
        # Get value 2 idx dictionary
        v2idx = get_dict_val2idx(A=d_minmax['min_r'])
        # get the most centered ones with longest distance w.r.t outliners 
        # for each index in d_idx[val], get the aver_score, select the ones with higest aver_mean
        d_res = get_topk_txt_within_idx(d_minmax, v2idx, topk, key='avg_r')
        #print(d_minmax['min_r'])
        a = []
        for l in label_lst:
            a += text_dict[l]
        text_dict[label1] = a
        # visual_res(d_res, text_dict, label1) #TODO rewrite text_dict required to have key 'combined'
        sorted_outliners = [x[0] for x in sorted([(key, count) for key, count in cnt.items()], key=lambda x: -x[1])]
        print(sorted_outliners)
        d_txt4def = {}
        kth_smallest_index = np.argpartition(d_minmax['avg_r'], topk+3)[:topk+3]
        d_txt4def = txt_collect(d_res, kth_smallest_index, d_txt4def, label1)
        samples = [text_dict[label1][idx] for idx in d_txt4def[label1]]
        return samples


if __name__=="__main__":

    # https://www.tizi365.com/topic/10092.html
    model_local_path ='/Users/jiayixian/projects/llm/llm/models/bge-large-en-v1.5'
    model = SentenceTransformer(model_local_path)
    label = ""
    data_path = "/Users/jiayixian/projects/llm/data/f{label}.xlsx" 

    df = pd.read_excel(data_path)
    label_col = 'my_label'
    col = 'utterance'

    guides_lst = txt2lst("/Users/jiayixian/projects/llm/llm/due_data.txt")

    top_k_labels = df[label_col].value_counts().head(8).index.tolist() # Series
    print(top_k_labels)

    label_lst = top_k_labels

    #mat_dict_combined, text_dict=get_txt_lst(model, label_lst, df)
    #label2st= get_label2st(label_lst, df, label_col='my_label', col='utterance')
    #mat_crx_dict = get_cross_matrix_dict(label_lst, label2st, mat_dict_combined['combined'])

    label_lst = ['statement_balance', '.total_balance', 'balance_x']
    # label_lst = ['balance_issue_incorrect', 'balance_issue_incorrect0']
    samples = generate_label2samples(model, df, label_lst, topk=1, Is_combined=True, path="", label_col='my_label', col='utterance') # text_dict, label2st, mat_dict_combined, mat_crx_dict, 

    import json
    label_joined_name = "_".join(label_lst)

    path = "/Users/jiayixian/projects/llm/llm/samples_dict.json"

    with open(path, "r") as f:
        label2samples_dict=json.load(f)
    f.close()

    if label_joined_name not in label2samples_dict:
        label2samples_dict[label_joined_name] = []
    label2samples_dict[label_joined_name].append(samples)

    with open(path, "w") as f:
        json.dump(label2samples_dict,f)
    f.close()