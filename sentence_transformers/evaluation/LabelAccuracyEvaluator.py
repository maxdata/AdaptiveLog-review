from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from ..util import batch_to_device
import os
import csv
import sklearn
import datetime
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,classification_report

logger = logging.getLogger(__name__)

class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        correct1 = 0
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

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)

        self.dataloader.collate_fn = model.smart_batching_collate

        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            total += prediction.size(0)


            # prediction = -1.0 * prediction
            # _, indices = torch.sort(prediction, dim=1)
            # _, indices = torch.sort(indices, dim=1)
            # correct1 += indices[torch.arange(label_ids.size(0)),label_ids].le(0).sum().item()
            # correct3 += indices[torch.arange(label_ids.size(0)),label_ids].le(1).sum().item()
            # correct5 += indices[torch.arange(label_ids.size(0)),label_ids].le(2).sum().item()


            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            y_pred.extend(torch.argmax(prediction, dim=1).cpu().numpy().tolist())
            y_true.extend(label_ids.cpu().numpy().tolist())



        print('macro_f1->>>',f1_score(y_true, y_pred, average='macro'))
        print('micro_f1->>>',f1_score(y_true, y_pred, average='micro'))
        print('weight_f1->>>',f1_score(y_true, y_pred, average='weighted'))

        # print(sklearn.metrics.classification_report(y_true, y_pred))
        # print(sklearn.metrics.confusion_matrix(y_true, y_pred).ravel())
        # print('P->>>', precision_score(y_true, y_pred))
        # print('R->>>', recall_score(y_true, y_pred))
        # print('F1->>>', f1_score(y_true, y_pred))

        # accuracy1 = correct1/total
        # accuracy3 = correct3/total
        # accuracy5 = correct5/total
        # print('recall 1 2 3',accuracy1,accuracy3,accuracy5)

        accuracy = correct / total
        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))



        # accuracy = correct/total
        # logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        # if output_path is not None and self.write_csv:
        #     csv_path = os.path.join(output_path, self.csv_file)
        #     if not os.path.isfile(csv_path):
        #         with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
        #             writer = csv.writer(f)
        #             writer.writerow(self.csv_headers)
        #             writer.writerow([epoch, steps, accuracy])
        #     else:
        #         with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
        #             writer = csv.writer(f)
        #             writer.writerow([epoch, steps, accuracy])

        return accuracy
