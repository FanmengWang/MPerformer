# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List


logger = logging.getLogger(__name__)


@register_model("MPerformer")
class MPerformerModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        
        self.bond_vocab_size = 5  # 0:no bound, 1:single bound, 2:double bound, 3:triple bound, 4:aromatic bond, 5:other_bond, 6:pad token
        self.atom_H_vocab_size = 4
        self.atom_charge_vocab_size = 3 
        
        self.activation_fn = args.activation_fn
        
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        
        self._num_updates = None
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
            
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.pair2coord_proj = NonLinearHead(args.encoder_attention_heads, 1, args.activation_fn)
                                
        self.classification_heads = nn.ModuleDict()
        
        self.atom_H_head = Atom_H_Head2(
            input_dim=args.encoder_embed_dim,
            inner_dim=args.encoder_embed_dim,
            num_classes=self.atom_H_vocab_size,
            activation_fn=args.pooler_activation_fn,
            pooler_dropout=args.pooler_dropout,
        )
        
        self.bond_head = BondHead2(args.encoder_attention_heads, args.activation_fn, self.bond_vocab_size)
        
        self.atom_charge_reg_head = PredictHead2(self.atom_charge_vocab_size, 1, args.activation_fn)

        self.atom_H_reg_head = PredictHead2(self.atom_H_vocab_size, 1, args.activation_fn)
        
        self.atom_bond_reg_head = PredictHead2(self.bond_vocab_size, 1, args.activation_fn)
        
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):  
                
        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        
        if not padding_mask.any():
            padding_mask = None
            
        x = self.embed_tokens(src_tokens)
                
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
                
        encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
                
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_coord = None
        encoder_distance = None
        encoder_bond = None
        activation_fn = utils.get_activation_fn(self.activation_fn)

        # logits            
        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
            pred_atom_charge = self.atom_charge_reg_head(logits.clone().detach())
        
        # logits_atom_H
        logits_atom_H = self.atom_H_head(encoder_rep)
        pred_atom_H = self.atom_H_reg_head(logits_atom_H.clone().detach())
                
        # encoder_coord
        coords_emb = src_coord
        if padding_mask is not None:
            atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                -1, 1, 1, 1
            )
        else:
            atom_num = src_coord.shape[1] - 1
        delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
        attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
        coord_update = delta_pos / atom_num * attn_probs
        coord_update = torch.sum(coord_update, dim=2)
        encoder_coord = coords_emb + coord_update
            
        # encoder_bond
        encoder_bond = self.bond_head(encoder_pair_rep)
        pred_atom_bond = self.atom_bond_reg_head(encoder_bond.clone().detach())
        
        return (
            logits,
            logits_atom_H, 
            encoder_distance,
            encoder_coord,
            pred_atom_charge,
            pred_atom_H,
            pred_atom_bond,
            x_norm,
            delta_encoder_pair_rep_norm,
            encoder_bond
        )      
        
    
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class PredictHead(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)

        self.linear2 = nn.Linear(input_dim, output_dim)
        
        self.activation_fn = utils.get_activation_fn(activation_fn)
        
    def forward(self, features, **kwargs):
        x = features
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x



class PredictHead2(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, input_dim)
        self.linear4 = nn.Linear(input_dim, input_dim)
        self.linear5 = nn.Linear(input_dim, output_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        
    def forward(self, features, **kwargs):
        x = features
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        x = self.activation_fn(x)
        x = self.linear3(x)
        x = self.activation_fn(x)
        x = self.linear4(x)
        x = self.activation_fn(x)
        x = self.linear5(x)
        return x



class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dense2 = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


class Atom_H_Head2(nn.Module):
    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dense2 = nn.Linear(input_dim, inner_dim)
        self.dense3 = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.dense3(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class BondHead2(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
        bond_vocab_size,
    ):
        super().__init__()
        self.bond_vocab_size = bond_vocab_size
        self.dense1 = nn.Linear(heads, heads)
        self.dense2 = nn.Linear(heads, heads)
        self.dense3 = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, bond_vocab_size)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        
        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        
        x = self.dense3(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        
        x = self.out_proj(x).view(bsz, seq_len, seq_len, self.bond_vocab_size)
        x = (x + x.transpose(1, 2)) * 0.5
        return x    
    

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("MPerformer", "MPerformer")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)


@register_model_architecture("MPerformer", "MPerformer_base")
def MPerformer_base_architecture(args):
    base_architecture(args)
