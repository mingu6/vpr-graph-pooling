from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.Dropout(p=0.5))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalized_coordinates(height, width):
    """ Normalized feature map coordinates"""
    gridH, gridW = torch.meshgrid(torch.arange(height),
                                    torch.arange(width))
    gridH = (gridH - height // 2) / (height * 0.7)
    gridW = (gridW - width // 2) / (width * 0.7)
    coords = torch.stack((gridW, gridH))
    return coords


def cartesian_to_polar(coords):
    r = coords.norm(dim=0)
    theta = torch.atan2(coords[1, ...], coords[0, ...]) / np.pi
    polar = torch.stack((r, theta))
    return polar


def polar_offsets(features_shape):
    """ for a feature map, return the relative offset between features """
    N, _, height, width = features_shape
    norm_coords = normalized_coordinates(height, width).reshape(2, -1)
    coords_expand = norm_coords.unsqueeze(2) - norm_coords.unsqueeze(1)
    # polar = cartesian_to_polar(coords_expand)
    polar = coords_expand
    return polar.unsqueeze(0)


class GCNUpdate(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features, adj):
        features = torch.matmul(adj, features)
        features = self.linear(features)
        features = F.relu(features)
        return features

class Attention(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.mlp = MLP([in_features, hidden_features, in_features])

    def forward(self, features):
        features_attn = self.mlp(features.transpose(-1, 1))
        attn = torch.matmul(features_attn.transpose(1, 2), features_attn)
        attn = F.softmax(attn, dim=-1)
        return attn


class PositionAdjacency(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(2))
        self.logChol = nn.Parameter(torch.ones(3) * -2.3)

    def forward(self, offsets):
        # construct precision matrix
        ind = torch.tril_indices(row=2, col=2, offset=0).to(offsets.device)
        chol = torch.zeros((2, 2)).to(offsets.device)
        # construct the cholesky decomp from unconstrained parameters
        chol[ind[0], ind[1]] = self.logChol
        chol[range(len(chol)), range(len(chol))] = torch.exp(chol.diag())
        prec = torch.matmul(chol, chol.t())
        # apply Gaussian kernel
        diff = offsets - self.mean.reshape(1, 2, 1, 1)
        kernel = torch.matmul(diff.transpose(1, -1), prec.transpose(0, 1)).transpose(-1, 1)
        kernel = torch.exp(-(kernel * diff).sum(dim=1))
        # normalize matrix
        kernel = F.normalize(kernel, p=1, dim=-1)
        # print("mean", self.mean, "logChol", self.logChol)
        return kernel


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.GC_a = GCNUpdate(in_features, out_features)
        self.GC_r = GCNUpdate(in_features, out_features)
        self.linear = nn.Linear(out_features * 2, out_features)
        self.attn = Attention(in_features, 2 * in_features)
        self.pos_adj = PositionAdjacency()

    def forward(self, features, offsets):
        # positional adjacency matrix
        adj_r = self.pos_adj(offsets)
        features_r = self.GC_r(features, adj_r)
        # attentional adjacency matrix
        adj_a = self.attn(features)
        features_a = self.GC_a(features, adj_a)
        # combine
        features = self.linear(torch.cat((features_a, features_r), dim=-1))
        return features

class VPRGNN(nn.Module):
    def __init__(self, in_feature_dim: int, out_feature_dim: int, num_layers: int):
        super().__init__()
        self.encoder = MLP([in_feature_dim, 3 * in_feature_dim, out_feature_dim])
        self.layers = nn.ModuleList([
            GCNLayer(out_feature_dim, out_feature_dim)
            for _ in range(num_layers)])

    def forward(self, features):
        N, C, _, _ = features.shape
        offsets = polar_offsets(features.shape).to(features.device)
        features = self.encoder(features.reshape(N, C, -1))
        features = features.transpose(1, -1)
        for layer in self.layers:
            features = layer(features, offsets)
        return features

class AttnPooling(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp = MLP([feature_dim, 2 * feature_dim, feature_dim])
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features):
        features_attn = self.mlp(features.transpose(-1, 1))
        features_attn = self.linear(features_attn.transpose(-1, 1))
        features = torch.sum(F.softmax(features_attn, dim=1) * features, dim=1)
        return features


class GraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = VPRGNN(
            config['descriptor_dim'], 2 * config['descriptor_dim'],
            config['GNN_layers'])
        self.pool = AttnPooling(2 * config['descriptor_dim'])

        # TO DO: load saved checkpoint?

    def forward(self, feature_map):
        feature_map = F.dropout(feature_map, p=0.5)
        features = self.gnn(feature_map)
        # graph aggregation to form final descriptor
        agg_desc =  self.pool(features)
        return F.normalize(agg_desc, p=2, dim=1)
