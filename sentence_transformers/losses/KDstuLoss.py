import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformerKD import SentenceTransformerKD
import logging
from sentence_transformers import util
import math
import os

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


class BertVLMHingeHead(nn.Module):
    def __init__(self, config, a=768, b=768):
        super().__init__()
        self.dense = nn.Linear(a, b)
        self.layer_norm = BertLayerNorm(b, eps=config.layer_norm_eps)

        # self.decoder = nn.Linear(b, b, bias=True)
        self.decoder = nn.Linear(312, b, bias=True)

    def forward(self, features, **kwargs):
        # x = self.dense(features)
        # x = gelu(x)
        # x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        # x = self.decoder(x)
        x = self.decoder(features)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x


class KDstuLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformerKD,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100)):
        super(KDstuLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.mlm = nn.Linear(sentence_embedding_dimension, len(self.model._first_module().tokenizer))
        self.level = nn.Linear(sentence_embedding_dimension, self.num_labels)
        self.loss_fct = loss_fct

        self.student_head = BertVLMHingeHead(self.model._first_module().auto_model.config, 768, 768).to(
            self.model._target_device)
        # self.teacher_head = BertVLMHingeHead(self.model._first_module().auto_model.config, 768, 768).to(
        #     self.model._target_device)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: [Dict, Tensor]):
        se = self.model(sentence_features[0])
        log_emb = se["sentence_embedding"]

        log_stu = self.student_head(log_emb)

        reps = se['token_embeddings']
        rep_a = reps
        mlm = self.mlm(rep_a)
        level = self.level(rep_a)
        teacher_embedding = labels["teacher_embedding"].to(self.model._target_device)
        # log_tea = self.teacher_head(teacher_embedding)
        log_tea = teacher_embedding

        distill_loss = util.contrastive_loss_item(log_stu, log_tea)

        mlm_label = labels["mlm_label"].to(self.model._target_device)
        level_label = labels["level_label"].to(self.model._target_device)
        loss_level = self.loss_fct(level.view(-1, self.num_labels), level_label.view(-1))
        loss_mlm = self.loss_fct(mlm.view(-1, len(self.model._first_module().tokenizer)), mlm_label.view(-1))
        loss = loss_level + loss_mlm + distill_loss
        return loss
