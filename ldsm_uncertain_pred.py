import argparse
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
from ldsm_uncertain_prob import calculate_var
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


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-test_data", "--test_data", type=str, help="train dataset")

    args.add_argument("-dev_data", "--dev_data", type=str, help="dev dataset")
    args.add_argument("-model_path", "--model_path", type=str, help="the path of the smaller model")

    args.add_argument("-out_simple_path", "--out_simple_path", type=str, help="Folder name to save the simple samples.")
    args.add_argument("-out_hard_path", "--out_hard_path", type=str, help="Folder name to save the hard samples.")

    args = args.parse_args()
    return args



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

def divide_hardsamples(args):

    x = read_json(args.test_data)
    test_samples = []
    for i in range(len(x)):
        test_samples.append(InputExample(texts=[x[i][0][0],x[i][0][1]], label=x[i][1]))

    dev_dataloader = DataLoader(test_samples, shuffle=False, batch_size=256)
    print(len(dev_dataloader))
    dev_dataloader.collate_fn = smart_batching_collate

    softmodel = torch.load(args.model_path).to('cuda')
    softmodel.train()
    var = []
    dis_label = []
    save_result = []

    raw_label = []

    dis_list = []
    all_dis = []

    a = dev_dataloader.dataset
    aa = []
    for item in a:
        aa.append([item.texts,item.label])


    for step, batch in enumerate(dev_dataloader):
        print(step)

        features, label_ids = batch
        for idx in range(len(features)):
            features[idx] = batch_to_device(features[idx], 'cuda')
        label_ids = label_ids.to('cuda')
        value = []
        for step, batch in enumerate(dev_dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], 'cuda')
            label_ids = label_ids.to('cuda')
            value = []
            for i in range(10):
                with torch.no_grad():
                    _, prediction = softmodel(features, labels=None)
                    certain = torch.softmax(prediction, dim=1)
                    select = certain.gather(1, label_ids.unsqueeze(1)).squeeze()
                    value.append(select)
            concat = torch.cat(value, dim=0).view(10, -1)
            var += concat.var(dim=0).tolist()
            raw_label += label_ids.tolist()
            dis_label += torch.argmax(prediction, dim=1).eq(label_ids).int().tolist()

            raw_data = concat.T

            data_mean = raw_data.mean(dim=1).unsqueeze(1).repeat(1, 10)
            dis = torch.abs(raw_data - data_mean)

            var_value  = (dis < calculate_var(args.dev_data,args.model_path)).sum(dim=1)
            prob = (args.prior + var_value ) / 11
            prob = prob.tolist()
    result_simple = []
    result_hard = []
    for i in range(len(prob)):
        if prob[i] < 0.5:
            result_simple.append(aa[i])
        else:
            result_hard.append(aa[i])
    save_json(result_simple,args.simple_path)
    save_json(result_hard,args.hard_path)


if __name__ == '__main__':
    args = parse_args()
    divide_hardsamples(args)