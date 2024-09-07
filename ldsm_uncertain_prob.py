#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
from torch.utils.data import DataLoader
import torch
from torch import Tensor, device
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,LabelAccuracyEvaluator,LabelAccSave
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import json
import random
import os
import sys
# sys.path.append("/home/mlp/sbert")


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
random.seed(1)

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)

# file = '../log_description.txt'
file = '../log/hw_switch_desc.json'
# file = '../log/h3c_security.json'
# file = '../log/cs_switch.json'


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

data_old = read_json(file)
data = []
for item in data_old:
    if item[1] != '' and item[1] != '-':
        data.append(item)
random.shuffle(data)
x = []
y = []
for i in range(len(data)):
    data[i][0] = data[i][0].replace('\\"','')
    x.append([data[i],1])
    y.append(1)
    if i != len(data) - 1:
        neg_causes = data[i + 1][1]
    else:
        neg_causes = data[0][1]
    x.append([[data[i][0],neg_causes],0])
    y.append(0)

random.shuffle(x)
# x = x[:4000]

model_name = 'bert-base-uncased'

train_size = int(len(x) * 0.6)
dev_size = int(len(x) * 0.8)

# train_size = 10000
# dev_size = 13000

def tokenize(texts):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    output = {}
    if isinstance(texts[0], str):
        to_tokenize = [texts]
    elif isinstance(texts[0], dict):
        to_tokenize = []
        output['text_keys'] = []
        for lookup in texts:
            text_key, text = next(iter(lookup.items()))
            to_tokenize.append(text)
            output['text_keys'].append(text_key)
        to_tokenize = [to_tokenize]
    else:
        batch1, batch2 = [], []
        for text_tuple in texts:
            batch1.append(text_tuple[0])
            batch2.append(text_tuple[1])
        to_tokenize = [batch1, batch2]

    # strip
    to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

    # Lowercase
    if True:
        to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

    output.update(tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt",
                            max_length=512))
    return output


def smart_batching_collate(batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
    Here, batch is a list of tuples: [(tokens, label), ...]

    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    num_texts = len(batch[0].texts)
    texts = [[] for _ in range(num_texts)]
    labels = []

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text)
        labels.append(example.label)

    labels = torch.tensor(labels)

    sentence_features = []
    for idx in range(num_texts):
        tokenized = tokenize(texts[idx])
        sentence_features.append(tokenized)

    # token分类
    # labels = labels[:,:sentence_features[0]['input_ids'].size(1)]

    return sentence_features, labels

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def calculate_var(data_path,model_path):
    data = read_json(data_path)
    dev_samples = []
    for i in range(len(data)):
        dev_samples.append(InputExample(texts=[x[i][0][0],x[i][0][1]],label=x[i][1]))

    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=256)

    a = dev_dataloader.dataset
    aa = []
    for item in a:
        aa.append([item.texts,item.label])


    print(len(dev_dataloader))
    dev_dataloader.collate_fn = smart_batching_collate

    softmodel = torch.load(model_path).to('cuda')
    softmodel.train()
    times = []
    correct = []
    error = []
    result = []


    var = []
    dis_label = []
    save_result = []

    raw_label = []


    dis_list = []
    all_dis = []
    for step, batch in enumerate(dev_dataloader):
        features, label_ids = batch
        for idx in range(len(features)):
            features[idx] = batch_to_device(features[idx], 'cuda')
        label_ids = label_ids.to('cuda')
        value = []
        for i in range(10):
            with torch.no_grad():
                _, prediction = softmodel(features, labels=None)
                certain = torch.softmax(prediction,dim=1)
                select = certain.gather(1, label_ids.unsqueeze(1)).squeeze()
                value.append(select)
        concat = torch.cat(value,dim=0).view(10,-1)
        var += concat.var(dim = 0).tolist()
        raw_label += label_ids.tolist()
        dis_label += torch.argmax(prediction, dim=1).eq(label_ids).int().tolist()


        raw_data = concat.T

        data_mean = raw_data.mean(dim=1).unsqueeze(1).repeat(1,10)
        dis = torch.abs(raw_data-data_mean)
        all_dis += dis.tolist()
        # all_dis += raw_data.tolist()
        dis_list += dis.var(dim = 1).tolist()

    for i in range(len(var)):
        save_result.append([var[i], dis_label[i]])
    # save_json(save_result,'ldsm_hwrouters_uncertain_knowlog.json')



    task_split_result = []
    for i in range(len(var)):
        task_split_result.append([aa[i],dis_label[i]])
    # save_json(task_split_result,'ldsm_hwrouters_tasksplit.json')

    all_error = []
    err = []
    core = []
    # for i in range(len(var)):
    #     if dis_label[i] == 0:
    #         all_error.append(aa[i])
    #         err.append(var[i])
    #     else:
    #         core.append(var[i])

    err_data = []
    corr_data = []
    for i in range(len(var)):
        if dis_label[i] == 0:
            err.append(dis_list[i])
            err_data.append(all_dis[i])
        else:
            corr_data.append(all_dis[i])
            core.append(dis_list[i])

    print(sum(err)/len(err))
    print(sum(core)/len(core))
    # print(err_data)
    # print(corr_data)

    return sum(core)/len(core)
