import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import tqdm
import pickle

# Define the MultiStageModel class
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal=False, use_graph=True, **graph_args):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, causal)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, causal)) for s in range(num_stages-1)])
        self.use_graph = use_graph
        if use_graph:
            self.graph_learner = TaskGraphLearner(**graph_args)

    def forward(self, x, mask):
        out, out_app = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        outputs_app = out_app.unsqueeze(0)
        for s in self.stages:
            out, out_app = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            outputs_app = torch.cat((outputs_app, out_app.unsqueeze(0)), dim=0)
        return outputs, outputs_app

# Define the ProbabilityProgressFusionModel class
class ProbabilityProgressFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(ProbabilityProgressFusionModel, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv1d(num_classes*2, num_classes, 1)

    def forward(self, in_cls, in_prg):
        ### in_cls: batch_size x num_classes x T
        ### in_prg: batch_size x num_classes x T
        # Concatenate classification and progress inputs
        input_concat = torch.cat((in_cls, in_prg), dim=1)
        out = self.conv(input_concat)
        return out

# Define the TaskGraphLearner class
class TaskGraphLearner(nn.Module):
    def __init__(self, init_graph_path, learnable=False, reg_weight=0.01, eta=0.01):
        super(TaskGraphLearner, self).__init__()
        with open(init_graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        matrix_pre, matrix_suc = self.graph['matrix_pre'], self.graph['matrix_suc']
        self.matrix_pre = nn.Parameter(torch.from_numpy(matrix_pre).float(), requires_grad=learnable)
        self.matrix_suc = nn.Parameter(torch.from_numpy(matrix_suc).float(), requires_grad=learnable)
        self.learnable = learnable
        if learnable:
            self.matrix_pre_original = nn.Parameter(self.matrix_pre, requires_grad=False)
            self.matrix_suc_original = nn.Parameter(self.matrix_suc, requires_grad=False)
        self.reg_weight = reg_weight
        self.eta = eta

    def forward(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        graph_loss = ((alpha_pre + alpha_suc) * action_prob).mean()
        if self.learnable:
            regularization = torch.mean((self.matrix_pre - self.matrix_pre_original) ** 2)
            return graph_loss + self.reg_weight * regularization
        return graph_loss

    def inference(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        logits = cls - self.eta * (alpha_pre + alpha_suc)
        return logits

# Define the SingleStageModel class
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal=causal)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        ### Action Progress Prediction (APP) module
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1, batch_first=True, bidirectional=not causal)
        self.conv_app = nn.Conv1d(num_f_maps*2 if not causal else num_f_maps*2 if not causal else num_f_maps, num_classes, 1)
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        prob_out = self.conv_out(out) * mask[:, 0:1, :]
        progress_out, _ = self.gru_app(out.permute(0, 2, 1))
        progress_out = progress_out.permute(0, 2, 1)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]
        out = self.prob_fusion(prob_out, progress_out)
        out = out * mask[:, 0:1, :]
        return out, progress_out

# Define the DilatedResidualLayer class
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, filter_size=3, causal=False):
        super(DilatedResidualLayer, self).__init__()
        self.causal = causal
        self.dilation = dilation
        padding = int(dilation * (filter_size-1) / 2)
        if causal:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding*2, padding_mode='replicate', dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, filter_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        if self.causal:
            out = out[..., :-self.dilation*2]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
