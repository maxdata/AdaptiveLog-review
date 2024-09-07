#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformerEnhance import SentenceTransformerEnhance
import logging
from sentence_transformers import util
import math
import os
from .. import util

logger = logging.getLogger(__name__)
class EnhanceLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformerEnhance,
                 sentence_embedding_dimension: int,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100),scale: float = 20.0, similarity_fct = util.cos_sim):
        super(EnhanceLoss, self).__init__()
        self.model = model
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.mlm = nn.Linear(sentence_embedding_dimension, len(self.model._first_module().tokenizer))
        self.param = nn.Linear(sentence_embedding_dimension, len(self.model._first_module().tokenizer))
        self.loss_fct = loss_fct

        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: [Dict, Tensor]):
        reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = reps[1]
        mlm = self.mlm(embeddings_a)
        param = self.param(embeddings_a)
        mlm_label = labels["mlm_label"].to(self.model._target_device)
        param_label = labels["param_label"].to(self.model._target_device)
        loss_mlm = self.loss_fct(mlm.view(-1, len(self.model._first_module().tokenizer)), mlm_label.view(-1))
        loss_param = self.loss_fct(param.view(-1, len(self.model._first_module().tokenizer)), param_label.view(-1))

        se_a = embeddings_a[:,0,:]
        se_b = embeddings_b[:,0,:]

        scores = self.similarity_fct(se_a, se_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]
        return loss_mlm + loss_param+ self.cross_entropy_loss(scores, labels)
