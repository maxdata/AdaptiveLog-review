import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformerKD import SentenceTransformerKD
import logging

logger = logging.getLogger(__name__)


class KDmlmLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformerKD,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100)):
        super(KDmlmLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.mlm = nn.Linear(sentence_embedding_dimension, len(self.model._first_module().tokenizer))
        self.match = nn.Linear(sentence_embedding_dimension,2)
        self.level = nn.Linear(sentence_embedding_dimension,self.num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: [Dict,Tensor]):
        reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        rep_a = reps[0]
        rep_cls = rep_a[:,0,:]
        mlm = self.mlm(rep_a)
        level = self.level(rep_a)
        match = self.match(rep_cls)
        mlm_label = labels["mlm_label"].to(self.model._target_device)
        match_label = labels["match_label"].to(self.model._target_device)
        level_label = labels["level_label"].to(self.model._target_device)
        loss_match = self.loss_fct(match, match_label.view(-1))
        loss_level = self.loss_fct(level.view(-1, self.num_labels), level_label.view(-1))
        loss_mlm = self.loss_fct(mlm.view(-1, len(self.model._first_module().tokenizer)), mlm_label.view(-1))
        loss = loss_match + loss_level + loss_mlm
        return loss

