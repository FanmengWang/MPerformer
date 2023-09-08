# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.data import Dictionary
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import torch.distributed as dist


class MultiClassFocalLossWithAlpha(torch.nn.Module):
    def __init__(self, alpha=[0.51, 0.002, 0.488], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  
        log_softmax = torch.log_softmax(pred, dim=1) 
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  
        logpt = logpt.view(-1)  
        ce_loss = -logpt  
        pt = torch.exp(logpt)  
        
        alpha = alpha.to(ce_loss.device)
        
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss 
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss



@register_loss("finetune_focal_loss_all_constraint_infer")
class FinetuneFocalLossAllConstraintInfer(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.bond_pad_idx = 6
        self.atom_H_pad_idx = 6
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
           
             
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(
            **sample["net_input"],
            classification_head_name=self.args.classification_head_name,
        )
                        
        logging_output = {
            "bsz": sample['target']['tokens_target'].size(0),
            "seq_len": sample['target']['tokens_target'].size(1) * sample['target']['tokens_target'].size(0),
            "id_name": sample['id_name']
        }
        
        logging_output["logit_output"] = net_output[0].data
        logging_output["target"] = sample['target']['tokens_target'].data
        logging_output["pred_atom_charge"] = net_output[4].data
        
        logging_output["logit_atom_H_output"] = net_output[1].data  
        logging_output["atom_H_target"] = sample['target']['atom_H_num_target'].data   
        logging_output["pred_atom_H"] = net_output[5].data
        
        logging_output["logit_bond_output"] = net_output[-1].data
        logging_output["bond_target"] = sample['target']['bond_target'].data
        logging_output["pred_atom_bond"] = net_output[6].data
        
        return 0, 1, logging_output
    

