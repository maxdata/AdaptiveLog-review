#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from .. import util
import os
import sys
import torch.nn.functional as F
class CodeMultipleNegativesRankingLoss2(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(CodeMultipleNegativesRankingLoss2, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]],labels: Tensor):
        # embeddings_a = self.model(sentence_features[0])['sentence_embedding']
        # for i in range(len(sentence_features)):

        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])
        # for key in sentence_features[1].keys():
        #     if sentence_features[1][key].size(1) != 512:
        #         padding = (0, 512 - sentence_features[1][key].size(1))
        #         padded_tensor = F.pad(sentence_features[1][key], padding, "constant", 0)
        #         sentence_features[1][key] = padded_tensor
        # all_sample = sentence_features[1]
        # for j in range(len(sentence_features)):
        #     for key in sentence_features[j].keys():
        #         if sentence_features[j][key].size(1) != 512:
        #             padding = (0, 512 - sentence_features[j][key].size(1))
        #             padded_tensor = F.pad(sentence_features[j][key], padding, "constant", 0)
        #             sentence_features[j][key] = padded_tensor
        #             all_sample[key] = torch.cat([all_sample[key],sentence_features[j][key]],0)

        embeddings_a = self.model(sentence_features[0])['sentence_embedding'].view(1,-1,768)
        embeddings_b = self.model(sentence_features[1])['sentence_embedding'].view(embeddings_a.size(0),-1,768)
        loss = 0
        for i in range(embeddings_a.size(0)):
            scores = self.similarity_fct(embeddings_a[i], embeddings_b[i]) * self.scale
            label = [1] + [0] * (embeddings_a.size(1) - 1)
            labels = torch.tensor(label, dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
            loss += self.cross_entropy_loss(scores, labels)

        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





