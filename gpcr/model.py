"""
Adapted from
https://github.com/xiiiz/CSCL-DTI
@inproceedings{lin2024cscl-dti,
title ={CSCL-DTI: predicting drug-target interaction through cross-view and self-supervised contrastive learning},
author ={Lin, Xuan and Zhang, Xi and Yu, Zu-Guo and Long, Yahui and Zeng, Xiangxiang and Yu, Philip S},
booktitle ={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
year ={2024}
}
"""

from gvp import GVP, GVPConvLayer, LayerNorm1
from torch_scatter import scatter_mean
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils import data
import sys
import os
from subword_nmt.apply_bpe import BPE
import codecs
import rdkit.Chem
from tqdm import tqdm
import torch
import pandas as pd
import torch_geometric
import torch_cluster
from constants import ATOM_VOCAB
from pdb_graph import _rbf, _normalize
from Bio.PDB import PDBParser, NeighborSearch
from torch_geometric.nn import global_add_pool, global_mean_pool
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import torch
from torch.utils import data
import pickle
import dgl
import deepchem
from subword_nmt.apply_bpe import BPE
import codecs
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from gvp import GVP, GVPConvLayer, LayerNorm1
from torch_scatter import scatter_mean
import numpy as np
import math
import copy
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from torch.nn.utils.weight_norm import weight_norm
import torch
from data_preprocess import Data_Encoder
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from dgl.nn.pytorch.conv import TAGConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class TAG(nn.Module):
    def __init__(self):
        super(TAG, self).__init__()
        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(TAGConv(31, 31, 2))
        self.pooling_protein = nn.Linear(31, 1)

    def forward(self, bg, num):
        feature_protein = bg.ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(bg, feature_protein))
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(bg, feature_protein)
        split_protein_rep = torch.split(protein_rep, num.tolist())
        max_len = 120 * 31
        flattened_proteins = []
        for matrix in split_protein_rep:
            flattened_matrix = matrix.view(-1)
            padding_len = max_len - len(flattened_matrix)
            if padding_len > 0:
                flattened_matrix = F.pad(flattened_matrix, (0, padding_len), mode='constant', value=0)
            flattened_proteins.append(flattened_matrix)
        final_rep = torch.stack(flattened_proteins)

        return final_rep


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 128))
        self.fc = nn.Linear(64 * 1 * 128, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, batch, atom_dim, hid_dim, out_dim=128, dropout=0.2):
        super(GCN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.batch = batch
        self.gc1 = GraphConvolution(atom_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, hid_dim)
        self.out = torch.nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = self.relu(x)
        x = self.out(x)
        x = self.relu(x)
        return self.dropout(x)

class Clip(nn.Module):
    def __init__(self, temperature=0.05):
        super(Clip, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2):
        N = features_1.size()[0]
        cat_features_1 = torch.cat([features_1, features_2])
        cat_features_2 = torch.cat([features_2, features_1])
        norm_1 = cat_features_1.norm(dim=1, keepdim=True).clamp(min=1e-9)
        norm_2 = cat_features_2.norm(dim=1, keepdim=True).clamp(min=1e-9)
        features_1 = cat_features_1 / norm_1
        features_2 = cat_features_2 / norm_2
        logit_scale = self.logit_scale.exp()
        logits_per_f1 = logit_scale * features_1 @ features_2.t()
        labels = torch.arange(2 * N).long().to(logits_per_f1.device)
        loss = self.loss_fun(logits_per_f1, labels) / 2

        return loss, logits_per_f1

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class DrugGVPModel(torch.nn.Module):
    def __init__(self,
                 node_in_dim=[66, 1], node_h_dim=[128, 64],
                 edge_in_dim=[16, 1], edge_h_dim=[32, 1],
                 num_layers=3, drop_rate=0.1
                 ):

        super(DrugGVPModel, self).__init__()
        self.W_v = torch.nn.Sequential(
            LayerNorm1(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = torch.nn.Sequential(
            LayerNorm1(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        self.layers = torch.nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = torch.nn.Sequential(
            LayerNorm1(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, xd):
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        out = global_add_pool(out, batch)

        return out


class Predictor(nn.Module):
    def __init__(self, hid_dim, device, dropout=0.2, atom_dim=75, protein_dim=35, batch=16, bias=True):
        super(Predictor, self).__init__()
        self.device = device
        self.fusionsize = hid_dim
        self.max_d = 50
        self.max_p = 545
        self.input_dim_drug = 23532
        self.input_dim_targe = 16693
        self.n_layer = 2
        self.emb_size = hid_dim
        self.dropout_rate = 0.1
        self.hidden_size = hid_dim
        self.intermediate_size = 512
        self.num_attention_heads = 4
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.batch_size = batch
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_targe, self.emb_size, self.max_p, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.tag = TAG()
        self.GVP = DrugGVPModel()
        self.drug_GCN = GCN(self.batch_size, atom_dim=atom_dim, hid_dim=128, out_dim=128, dropout=dropout)
        self.protein_GCN = GCN(self.batch_size, atom_dim=35, hid_dim=128, out_dim=128, dropout=dropout)
        self.cnn = CNNModel()
        self.clip = Clip()
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.dropout = dropout
        self.do = nn.Dropout(dropout)
        self.atom_dim = atom_dim
        self.decoder_1 = nn.Sequential(
            nn.Linear(self.max_d * self.emb_size, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hid_dim)
        )
        self.decoder_2 = nn.Sequential(
            nn.Linear(self.max_p * self.emb_size, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, hid_dim)
        )
        self.flatten = nn.Flatten()
        self.query_proj = nn.Linear(256, 256 * 2, bias=False)
        self.key_proj = nn.Linear(256, 256 * 2, bias=False)
        self.value_proj = nn.Linear(256, 256 * 2, bias=False)
        self.output_proj = nn.Linear(256 * 2, 256, bias=False)
        seq_dict_len = 545
        self.embedding_layer = nn.Embedding(seq_dict_len + 1, 128)
        self.linear2 = nn.Sequential(
            nn.Linear(120 * 31, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, atom_feature, adj, num_size, d_v, p_v, input_mask_d, input_mask_p, drug_graphs, protein_graphs,
                protein_num, node_feature, p_adj, res_size):
        num_size = num_size.size(0)
        res_node = node_feature.shape[1]
        node_feature = torch.zeros((self.batch_size, res_node, 35), device=node_feature.device)

        ex_d_mask = input_mask_d.unsqueeze(1).unsqueeze(2)
        ex_p_mask = input_mask_p.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        d_emb = self.demb(d_v)
        p_emb = self.pemb(p_v)
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
        d1_trans_fts = d_encoded_layers.view(num_size, -1)
        d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)
        p1_trans_fts = p_encoded_layers.view(num_size, -1)
        p1_trans_fts_layer1 = self.decoder_2(p1_trans_fts)

        cmp_gnn_out = self.drug_GCN(atom_feature, adj)
        is_max = False
        if is_max:
            cmp_gnn_out = cmp_gnn_out.max(dim=1)[0]
        else:
            cmp_gnn_out = cmp_gnn_out.mean(dim=1)
        prot_conv_out = self.tag(protein_graphs, protein_num)
        prot_conv_out = self.linear2(prot_conv_out)

        embeddings = []
        for i in range(len(drug_graphs)):
            g = drug_graphs[i]
            data = g[0]
            drug_embedding = self.GVP(data)
            embeddings.append(drug_embedding)
        drug_gvp_out = torch.cat(embeddings, dim=0)
        protein_gnn_out = self.protein_GCN(node_feature, p_adj)
        is_max = False
        if is_max:
            protein_gnn_out = protein_gnn_out.max(dim=1)[0]
        else:
            protein_gnn_out = protein_gnn_out.mean(dim=1)

        contrast_loss_d1, logits_per_d_f1 = self.clip(d1_trans_fts_layer1, cmp_gnn_out)
        contrast_loss_d2, logits_per_d_f2 = self.clip(d1_trans_fts_layer1, drug_gvp_out)
        contrast_loss_d3, logits_per_d_f3 = self.clip(cmp_gnn_out, drug_gvp_out)

        contrast_loss_p1, logits_per_p_f1 = self.clip(p1_trans_fts_layer1, prot_conv_out)
        contrast_loss_p2, logits_per_p_f2 = self.clip(p1_trans_fts_layer1, protein_gnn_out)
        contrast_loss_p3, logits_per_p_f3 = self.clip(prot_conv_out, protein_gnn_out)

        output1 = torch.cat((d1_trans_fts_layer1, cmp_gnn_out), dim=-1)
        drug_out = torch.cat((output1, drug_gvp_out), dim=-1)
        output2 = torch.cat((p1_trans_fts_layer1, prot_conv_out), dim=-1)
        protein_out = torch.cat((output2, protein_gnn_out), dim=-1)
        final_fts_cat = torch.cat((drug_out, protein_out), dim=-1)
        drug_loss = (contrast_loss_d1+contrast_loss_d2 + contrast_loss_d3)/3
        protein_loss = (contrast_loss_p1 + contrast_loss_p2 + contrast_loss_p3)/3
        contrast_loss = (drug_loss + protein_loss) /2

        torch.cuda.empty_cache()
        return self.out(final_fts_cat), contrast_loss

    def __call__(self, data, train=True):
        atom_feature, adj, num_size, d_v, p_v, input_mask_d, input_mask_p, drug_graphs, protein_graphs, protein_num, node_feature, p_adj, res_size, y = data
        Loss = nn.CrossEntropyLoss()
        if train:
            predicted_interaction, CR_loss = self.forward(atom_feature, adj, num_size, d_v, p_v, input_mask_d,
                                                          input_mask_p, drug_graphs, protein_graphs, protein_num,
                                                          node_feature, p_adj,
                                                          res_size)
            CE_loss = Loss(predicted_interaction, y)
            print('CR_loss:', CR_loss)
            print('CE_loss:', CE_loss)
            loss = 0.01 * CR_loss + 2.0 * CE_loss
            return loss
        else:
            predicted_interaction, CR_loss = self.forward(atom_feature, adj, num_size, d_v, p_v, input_mask_d,
                                                          input_mask_p, drug_graphs, protein_graphs, protein_num,
                                                          node_feature, p_adj,
                                                          res_size)
            correct_labels = y.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b = torch.LongTensor(1, 2)
        b = b.cuda()
        input_ids = input_ids.type_as(b)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)  # +注意力
        attention_output = self.output(self_output, input_tensor)  # +残差
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


MAX_PROTEIN_LEN = 545
MAX_DRUG_LEN = 50


def data_to_device(data, device):
    atoms_new, adjs_new, num_size_new, d_new, p_new, dmask_new, pmask_new, drug_graphs, protein_graphs, protein_num_new, node_feature_new, p_adj_new, res_size_new, label_new = data
    atoms_new = atoms_new.to(device)
    adjs_new = adjs_new.to(device)
    num_size_new = num_size_new.to(device)
    d_new = d_new.to(device)
    p_new = p_new.to(device)
    dmask_new = dmask_new.to(device)
    pmask_new = pmask_new.to(device)
    protein_graphs = protein_graphs.to(device)
    protein_num_new = torch.tensor(protein_num_new).to(device)
    res_size_new = res_size_new.to(device)
    p_adj_new = p_adj_new.to(device)
    node_feature_new = node_feature_new.to(device)
    label_new = label_new.to(device)
    drug_graphs = [
        [graph.to(device) for graph in batch]
        for batch in drug_graphs
    ]

    return (atoms_new, adjs_new, num_size_new, d_new, p_new, dmask_new, pmask_new, drug_graphs,
            protein_graphs, protein_num_new, node_feature_new, p_adj_new, res_size_new, label_new)


from transformers import AdamW, get_cosine_schedule_with_warmup


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, n_sample):
        self.model = model
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                            num_training_steps=n_sample // batch)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()

        for i, data_pack in enumerate(dataset):
            data_pack = data_to_device(data_pack, device)
            loss = self.model(data_pack)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model
    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for i, data_pack in enumerate(dataset):
                data_pack = data_to_device(data_pack, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                torch.cuda.empty_cache()
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall,Y

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
