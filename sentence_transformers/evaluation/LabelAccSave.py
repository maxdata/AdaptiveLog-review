from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from ..util import batch_to_device
import os
import csv
import sklearn
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import json
logger = logging.getLogger(__name__)
def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


import math


def calculate_variance(numbers):
    avg = sum(numbers) / len(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / len(numbers)
    return variance


def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)

class LabelAccSave(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        correct2 = 0
        correct3 = 0
        correct5 = 0
        y_pred = []
        y_true = []
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)

        a = self.dataloader.dataset
        aa = []
        for item in a:
            aa.append([item.texts,item.label])
        correct = []
        error = []
        self.dataloader.collate_fn = model.smart_batching_collate
        softmodel = torch.load('/home/mlp/adaptive/model/small_model.pt').to('cuda')
        # softmodel = torch.load('/home/mlp/adaptive/model/discriminate_model.pt').to('cuda')
        # softmodel.eval()
        softmodel.train()
        times = []

        result = []

        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            for i in range(10):
                with torch.no_grad():
                    # _, prediction = self.softmax_model(features, labels=None)
                    _, prediction = softmodel(features, labels=None)
                    # print(prediction)
                    # print(torch.softmax(prediction,dim=1))
                    # print(torch.softmax(prediction,dim=1))
                    times.append(torch.softmax(prediction,dim=1))




        #     total += prediction.size(0)
        #     correct2 += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        #
        # accuracy = correct2 / total
        # print(accuracy)
            if torch.argmax(prediction, dim=1).eq(label_ids).sum().item() == 1:
                correct.append([aa[step],torch.max(prediction, dim=1).values.item(),torch.max(prediction, dim=1).indices.item()])
                result.append(1)
            else:
                result.append(0)
                error.append([aa[step],torch.max(prediction, dim=1).values.item(),torch.max(prediction, dim=1).indices.item()])
        print(len(error))
        print(len(correct))
        # print(error[0])

        # for item in times:
        #     print(item[0].cpu().tolist())
        var = []
        for i in range(0, len(times), 10):
            line = []
            for j in range(10):
                line.append(times[i + j].cpu().tolist()[0][0])
            var.append(calculate_variance(line))
        print(var)
        sava_result = []
        for i in range(len(var)):
            sava_result.append([var[i],result[i]])
        save_json(sava_result,'ldsm_hwswitch_uncertain.json')
        # print(sum(var)/len(var))


        # save_json(error,'/home/mlp/adaptive/error.json')
        # save_json(correct,'/home/mlp/adaptive/correct.json')
        return 1
