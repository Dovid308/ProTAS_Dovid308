import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import copy
import numpy as np
import tqdm
import pickle
from utils.utils_paths import resolve_entry_paths
from  utils.eval import *
import json

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

class APPModule(nn.Module):
    def __init__(self, num_f_maps, num_classes, causal=False):
        super(APPModule, self).__init__()
        self.gru = nn.GRU(num_f_maps, num_f_maps, num_layers=1,
                          batch_first=True, bidirectional=not causal)
        self.conv = nn.Conv1d(num_f_maps * (1 if causal else 2), num_classes, 1)

    def forward(self, feature, mask):
        out, _ = self.gru(feature.permute(0, 2, 1))
        return self.conv(out.permute(0, 2, 1)) * mask[:, 0:1, :]

# Define the SingleStageModel class
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal=causal)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
        ### Action Progress Prediction (APP) module
        self.app = APPModule(num_f_maps, num_classes, causal)
      
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        prob_out = self.conv_out(out) * mask[:, 0:1, :]
        progress_out = self.app(out, mask)
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



from utils.asformer import Encoder, Decoder, exponential_descrease


class ASFormerPROTAS(nn.Module):
    """
    ASFormer + PROTAS: APP module e fusion su ogni stage.
    Paper: 1 encoder + 3 decoder, 9 layer ciascuno.

    #Opzione B — modifichi construct_window_mask in asformer.py per supportare la causal mask, e passi il flag fino in fondo. Questo replica correttamente il paper.
    #direi che devo ancora fare questo no 
    """
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps,
                 input_dim, num_classes, channel_masking_rate, causal=False,
                 use_graph=True, init_graph_path='', learnable=True):
        super(ASFormerPROTAS, self).__init__()
        num_stages = num_decoders + 1
 
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            num_classes,
            channel_masking_rate,
            att_type='sliding_att',
            alpha=1,
            causal=causal
        )
        
        self.decoders = nn.ModuleList([
            copy.deepcopy(
                Decoder(
                    num_layers,
                    r1,
                    r2,
                    num_f_maps,
                    num_classes,
                    num_classes,
                    att_type='sliding_att',
                    alpha=exponential_descrease(s),
                    causal=causal
                )
            )
            for s in range(num_decoders)
        ])
        self.app_modules = nn.ModuleList([
            APPModule(num_f_maps, num_classes, causal) for _ in range(num_stages)
        ])
        self.fusion_modules = nn.ModuleList([
            ProbabilityProgressFusionModel(num_classes) for _ in range(num_stages)
        ])
        self.use_graph = use_graph
        if use_graph:
            self.graph_learner = TaskGraphLearner(init_graph_path, learnable)
 
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        progress = self.app_modules[0](feature, mask)
        out = self.fusion_modules[0](out, progress) * mask[:, 0:1, :]
 
        outputs = out.unsqueeze(0)
        outputs_progress = progress.unsqueeze(0)
 
        for i, decoder in enumerate(self.decoders):
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :],
                                   feature * mask[:, 0:1, :], mask)
            progress = self.app_modules[i + 1](feature, mask)
            out = self.fusion_modules[i + 1](out, progress) * mask[:, 0:1, :]
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            outputs_progress = torch.cat((outputs_progress, progress.unsqueeze(0)), dim=0)
 
        return outputs, outputs_progress



class Pipeline:
    def __init__(self, model, num_classes, logger, save_dir, actions_dict, progress_lw=1, graph_lw=0.1, bg_class=['BG'], map_delimiter='|'):
        self.model = model  # ricevuto già costruito
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.progress_lw = progress_lw
        self.graph_lw = graph_lw
        self.logger = logger
        self.actions_dict = actions_dict
        self.save_dir = save_dir
        self.bg_class = bg_class
        self.map_delimiter = map_delimiter


    def train(self, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_acc = 0.0
        # percorsi fissi — nessun rolling da tracciare
        best_model_path = os.path.join(self.save_dir, "best.model")
        best_opt_path   = os.path.join(self.save_dir, "best.opt")

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_progress_loss = 0
            epoch_graph_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, batch_progress_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_progress_target, mask = batch_input.to(device), batch_target.to(device), batch_progress_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, progress_predictions = self.model(batch_input, mask)

                loss = 0
                progress_loss = 0
                for p, progress_p in zip(predictions, progress_predictions):
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:])
                    progress_loss += self.mse(progress_p, batch_progress_target).mean()

                loss += self.progress_lw * progress_loss
                epoch_progress_loss += self.progress_lw * progress_loss.item()

                graph_loss = self.model.graph_learner(predictions[-1], progress_predictions[-1])
                loss += self.graph_lw * graph_loss
                epoch_graph_loss += self.graph_lw * graph_loss.item()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            acc = float(correct) / total

            # --- best model save ---
            if acc > best_acc:
                best_acc = acc
                # sovrascrive direttamente best.model / best.opt (nomi fissi, nessuna pulizia necessaria)
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_opt_path)
                self.logger.info(f"  --> New best model saved: best.model  (train acc={acc:.4f}, epoch={epoch + 1})")
            # --------------------------------
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            print(f"Epoch {epoch} | Alloc: {alloc:.1f} MB | Reserved: {reserved:.1f} MB", flush=True)
            self.logger.info("[epoch %d]: epoch loss = %f, progress loss = %f, graph loss = %f, acc = %f" % (epoch + 1,
                                                              epoch_loss / len(batch_gen.list_of_examples),
                                                              epoch_progress_loss / len(batch_gen.list_of_examples),
                                                              epoch_graph_loss / len(batch_gen.list_of_examples),
                                                              acc))

        return best_model_path

  

    def predict(self, vid_list_file, device, sample_rate, feature_transpose=False, dataset_root=None, is_unified=False):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            # il modello sta sempre in exp_dir/best.model
            model_path = os.path.join(self.save_dir, "best.model")
            self.model.load_state_dict(torch.load(model_path))

            # le predizioni vanno nella sottocartella predict/
            predict_dir = os.path.join(self.save_dir, "predict")
            os.makedirs(predict_dir, exist_ok=True)

            bundle_path = f"{dataset_root}/splits/{vid_list_file}.bundle"
            file_ptr = open(bundle_path, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # OLD:
                # features = np.load(features_path + vid.split('.')[0] + '.npy')

                #  il path delle feature viene risolto dalla entry del bundle
                _, features_path, _ = resolve_entry_paths(dataset_root, vid, is_unified)
                features = np.load(features_path)

                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, progress_predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                _, predicted = torch.max(final_predictions.data, 1)

                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(self.actions_dict.keys())[list(self.actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(os.path.join(predict_dir, f_name), "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(self.map_delimiter.join(recognition))
                f_ptr.close()

    def predict_online(self, vid_list_file, device, sample_rate, feature_transpose=False, dataset_root=None, is_unified=False):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            # il modello sta sempre in save_dir/best.model
            model_path = os.path.join(self.save_dir, "best.model")
            self.model.load_state_dict(torch.load(model_path))

            # le predizioni vanno nella sottocartella predict/
            predict_dir = os.path.join(self.save_dir, "predict")
            os.makedirs(predict_dir, exist_ok=True)

            bundle_path = f"{dataset_root}/splits/{vid_list_file}.bundle"
            file_ptr = open(bundle_path, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # OLD:
                # features = np.load(features_path + vid.split('.')[0] + '.npy')

                #  il path delle feature viene risolto dalla entry del bundle
                _, features_path, _ = resolve_entry_paths(dataset_root, vid, is_unified)
                features = np.load(features_path)

                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                n_frames = input_x.shape[-1]
                recognition = []
                for frame_i in tqdm.tqdm(range(n_frames)):
                    curr_input_x = input_x[:, :, :frame_i+1]
                    predictions, progress_predictions = self.model(curr_input_x, torch.ones(curr_input_x.size(), device=device))
                    final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                    _, predicted = torch.max(final_predictions.data, 1)
                    predicted = predicted.squeeze(0)
                    recognition = np.concatenate((recognition, [list(self.actions_dict.keys())[list(self.actions_dict.values()).index(predicted[-1].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(os.path.join(predict_dir, f_name), "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(self.map_delimiter.join(recognition))
                f_ptr.close()
    
    def evaluate(self, vid_list_file, dataset_root=None, is_unified=False):
        # Bundle path risolto

        

        bundle_path = f"{dataset_root}/splits/{vid_list_file}.bundle"
        with open(bundle_path, 'r') as f:
            list_of_videos = f.read().split('\n')[:-1]

        overlap = [.1, .25, .5] 
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        correct = 0     
        total = 0      
        correct_wo_bg = 0
        total_wo_bg = 0 
        edit = 0       

        

        # le predizioni stanno nella sottocartella predict/
        predict_dir = os.path.join(self.save_dir, "predict")

        for vid in list_of_videos:
            if not vid.endswith('.txt'):
                vid = vid + '.txt'

            gt_path, _, _ = resolve_entry_paths(dataset_root, vid, is_unified)
            gt_content = gt_path.read_text().strip().splitlines()

            f_name = vid.split('/')[-1].split('.')[0]
            # MODIFICATO: legge da predict_dir invece che da result_dir
            recog_file = os.path.join(predict_dir, f_name)
            #inline
            with open(recog_file, 'r') as f:
                recog_content = f.read().split('\n')[1].split(self.map_delimiter)

            for i in range(len(gt_content)):
                if gt_content[i] not in self.bg_class:
                    total_wo_bg += 1
                    if gt_content[i] == recog_content[i]:
                        correct_wo_bg += 1

                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            # BUGFIX: edit_score calcolato una volta per video, non dentro il loop per frame
            edit += edit_score(recog_content, gt_content, bg_class=self.bg_class)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], self.bg_class)
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        acc = 100*float(correct)/total  
        acc_wo_bg = 100*float(correct_wo_bg)/total_wo_bg  
        edit = (1.0*edit)/len(list_of_videos)
        res_list = [acc, acc_wo_bg, edit]

        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
            f1 = 2.0 * (precision*recall) / (precision+recall)
            f1 = np.nan_to_num(f1)*100         
            res_list.append(f1)

        print("Result:", ' '.join(['{:.2f}'.format(r) for r in res_list]))
        result_metrics = {'Acc': acc,  'Acc-bg': acc_wo_bg, 'Edit': edit,
                          'F1@10': res_list[-3], 'F1@25': res_list[-2], 'F1@50': res_list[-1]}
        # MODIFICATO: JSON salvato in result_dir (expdir), non in predict_dir
        result_path = os.path.join(self.save_dir, vid_list_file + '.eval.json')
        with open(result_path, 'w') as fw:
            json.dump(result_metrics, fw, indent=4)
