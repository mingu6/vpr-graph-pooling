from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def MLP(channels: list, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                #layers.append(nn.Dropout(p=0.5))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalized_coordinates(features_shape):
    """ Normalized feature map coordinates"""
    N, _, height, width = features_shape
    gridH, gridW = torch.meshgrid(torch.arange(height),
                                    torch.arange(width))
    gridH = (gridH - height // 2) / (height * 0.7)
    gridW = (gridW - width // 2) / (width * 0.7)
    coords = torch.stack((gridW, gridH))
    return coords.unsqueeze(0).repeat(N, 1, 1, 1)


class CoordEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, coords):
        return self.encoder(coords.view(*coords.shape[:2], -1))


def attention(query, key, value, dropout, training):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    if dropout:
        prob = torch.nn.functional.dropout(prob, p=0.5, training=training)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.prob = []

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, True, self.training)
        # x, prob = attention(query, key, value, False, self.training)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        # dropout here?
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_layers)])

    def forward(self, desc):
        for layer in self.layers:
            layer.attn.prob = []
            delta = layer(desc, desc)
            desc = (desc + delta)
        return desc


class Linear(nn.Module):
    def __init__(self, feature_dim: int, num_hidden: int):
        super().__init__()
        hidden = nn.Linear(feature_dim, num_hidden)
        final = nn.Linear(num_hidden, feature_dim)
        self.layers = nn.Sequential(hidden, nn.ReLU(), final)

    def forward(self, features):
        return self.layers(features)

def attention1(query, key, value, training):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores = scores.mean(dim=1)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return prob

class AttentionAdjacency(nn.Module):
    """ Multi-head attention for learned adjacency matrix """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        prob = attention1(query, key, value, self.training)
        return prob


class GraphAggregation(nn.Module):
    def __init__(self, feature_dim: int, num_hidden: int):
        super().__init__()
        self.mlp = Linear(feature_dim, num_hidden)
        self.mlp_gate = Linear(feature_dim, num_hidden)
        self.mlp_agg = Linear(feature_dim, num_hidden)

    def forward(self, features):
        embedding = self.mlp(features.transpose(1, 2))
        gate = self.mlp_gate(features.transpose(1, 2))
        agg = F.glu(torch.cat((embedding, gate), dim=-1), dim=-1).sum(dim=1)
        agg_final = self.mlp_agg(agg)
        return agg_final


class GraphAggregation2(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.att = AttentionAdjacency(4, feature_dim)
        self.attn_wts = nn.Linear(feature_dim, 1)

    def forward(self, features):
        # compute node attention weights
        A = self.att(features, features, features)
        attn = torch.matmul(A, features.transpose(1, 2))
        attn = self.attn_wts(attn).transpose(1, 2)
        attn = F.softmax(attn, dim=1)
        # aggregate nodes with attention weights
        agg = torch.sum(features * attn, dim=2)
        return agg

class GraphAggregation3(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()

    def forward(self, features):
        # max pooling
        pooled, _ = torch.max(features, dim=2)
        return pooled

class AttentionalPropagation2(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, features):
        features = self.mlp(features)
        A = torch.matmul(features.transpose(1, 2), features)
        # topk selection?
        A = F.softmax(A, dim=-2)  # attention weights
        features = torch.matmul(features, A)
        return features

class AttentionalGNN2(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation2(feature_dim)
            for _ in range(num_layers)])

    def forward(self, desc):
        for layer in self.layers:
            delta = layer(desc)
            desc = (desc + delta)  # residual component
        return desc


def adjacency_position(coords1, coords2):
    """ computes polar coordinates and relative position pairwise """
    asdasd


class GraphPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_shape = feature_map.shape

        # normalize feature map coordinates 

        coords = normalized_coordinates(feature_map.shape)
        


class GraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cenc = CoordEncoder(
            config['descriptor_dim'], config['coord_encoder'])
        self.gnn = AttentionalGNN(
            config['descriptor_dim'], config['GNN_layers'])
        # self.gnn = AttentionalGNN2(
            # config['descriptor_dim'], config['GNN_layers'])
        # self.agg = GraphAggregation(
            # config['descriptor_dim'], config['agg_hidden']
        # )
        self.agg = GraphAggregation3(config['descriptor_dim'])
        # self.agg = GraphAggregation2(config['descriptor_dim'])

        # TO DO: load saved checkpoint?

    def forward(self, feature_map):
        feat_shape = feature_map.shape

        # normalize feature map coordinates 

        # coords = normalized_coordinates(feature_map.shape)
        # coords = coords.to(feature_map.device)

        # encode into feature space and combine

        # desc = feature_map.view(*feat_shape[:2], -1) + self.cenc(coords)
        desc = feature_map.view(*feat_shape[:2], -1)

        # Multi-layer Transformer

        desc = self.gnn(desc)

        # graph aggregation to form final descriptor
        # print("Allocated", torch.cuda.memory_allocated())
        # print("Cached", torch.cuda.memory_cached())
        agg_desc =  self.agg(desc)
        return F.normalize(agg_desc, p=2, dim=1)
