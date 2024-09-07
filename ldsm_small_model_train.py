import argparse
from torch.utils.data import DataLoader
import math
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


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-train_data", "--train_data", type=str, help="train dataset")

    args.add_argument("-dev_data", "--dev_data", type=str,help="dev dataset")


    args.add_argument("-pretrain_model", "--pretrain_model", type=str,
                      default="bert-base-uncased", help="the path of the pretrained model to finetune")


    args.add_argument("-epoch", "--epoch", type=int,
                      default=3, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=16, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default='ldsm_slm.pt', help="Folder name to save the models.")

    args = args.parse_args()
    return args


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

def train_model(args):
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    train_data = read_json(args.train_data)
    dev_data = read_json(args.dev_data)
    test_data = read_json(args.test_data)

    # load model
    model = SentenceTransformer(args.pretrain_model)

    # load dataset
    train_samples = []
    dev_samples = []
    test_samples = []
    for item in train_data:
        train_samples.append(InputExample(texts=[item[0][0], item[0][1]], label=item[1]))
    for item in dev_data:
        dev_samples.append(InputExample(texts=[item[0][0], item[0][1]], label=item[1]))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)

    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=2)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    small_model_path = args.small_model_path

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              evaluation_steps=10000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              save_task=True,
              small_model_path = small_model_path
              )


if __name__ == '__main__':
    args = parse_args()
    train_model(args)