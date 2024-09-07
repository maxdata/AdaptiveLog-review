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
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
# sys.path.append("/home/mlp/sbert")


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

    args.add_argument("-test_data", "--test_data", type=str,
                       help="path of test dataset")

    args.add_argument("-pretrain_model", "--pretrain_model", type=str,help="the path of the pretrained model to load")


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


def evaluate(args):
    data1 = read_json(args.test_data)
    data = data1
    test_samples = []

    for item in data:
        test_samples.append(InputExample(texts=[item[0][0][0],item[0][0][1]],label=item[0][1]))


    dev_dataloader = DataLoader(test_samples, shuffle=False, batch_size=256)
    print(len(dev_dataloader))
    dev_dataloader.collate_fn = smart_batching_collate

    softmodel = torch.load(args.model_path).to('cuda')

    total = 0
    correct = 0

    y_pred = []
    y_true = []
    pred_result = []
    for step, batch in enumerate(dev_dataloader):
        print(step)

        features, label_ids = batch
        for idx in range(len(features)):
            features[idx] = batch_to_device(features[idx], 'cuda')
        label_ids = label_ids.to('cuda')
        value = []
        with torch.no_grad():
            _, prediction = softmodel(features, labels=None)

        total += prediction.size(0)
        correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        pred_result += torch.argmax(prediction, dim=1).int().tolist()
        y_pred.extend(torch.argmax(prediction, dim=1).cpu().numpy().tolist())
        y_true.extend(label_ids.cpu().numpy().tolist())

    print('acc->>>', accuracy_score(y_true,y_pred))
    print('weight_f1->>>',f1_score(y_true, y_pred, average='weighted'))


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)