#!/usr/bin/env python3
"""
Command Example:
    
    python 09_bayesian_gnn_ensemble.py --lambda_corr 1.0

Descriptions (Eng / Kor):

    [critical options]
    --meta_csv        "add path to csv file with meta-data"
    --graphs_root    "add path that contains ligands.pt"
    --out_dir        "add where to save output results"

    [learning option defaults]
    --epoch 200 --batch_size 32 --lr 1e-3 --weight_decay 1e-5
    --dropout 0.1    "For Monte-Carlo based Bayesian Confidence calculation"
    --mc_T            "How many time you want to ramdomly re-try"

    [korean version]
    --meta_csv        "개별 리간드에서 추출한 feature가 담긴 pt파일 경로를 저장한 meta-data.csv"
    --graphs_root    "추출한 feature가 담긴 pt파일들이 모여있는 파일 경로"
    --out_dir        "어디에 결과를 저장할건지"

4개의 모델 GINEconv, GATv2, Transformer, GCN 을 앙상블로 구성했습니다.
각 모델 당 5개의 랜덤 Monte-Carlo Dropout을 통해, 추정의 불확실성을 정량측정하며,
이를 통해 통계적으로 베이지안에 근사한 Heuristic 기법을 구현합니다.
Affine calibration으로, 모델 예측값의 시스템적 편향을 보정합니다.
총 200번의 epoch 중 25회 이상 val loss가 개선되지 않으면 Early stop합니다.

"""

# -------------------------
# Library
# -------------------------

import os
import json
import time
import argparse
from pathlib import Path
import random
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

USE_PYG = False
try:
    from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, GINEConv, global_mean_pool, GlobalAttention
    from torch_geometric.nn.aggr import AttentionalAggregation
    USE_PYG = True
    print("[info] torch_geometric's GNN layers are available.")
except Exception:
    USE_PYG = False
    print("[warn] torch_geometric not found. Using a simple MLP-based model as a fallback.")

# -------------------------
# Meta-data and Graph Load
# -------------------------

def load_graphs_from_meta(meta_csv, graphs_root=None, target_cols=('gnina_affinity','vina_affinity'), replace_nan_with=None):
    df = pd.read_csv(meta_csv)
    if graphs_root is not None:
        df['graph_path'] = df['graph_path'].apply(lambda p: os.path.join(graphs_root, os.path.basename(p)) if not os.path.isabs(p) else p)
    data_list = []
    for _, row in df.iterrows():
        gpath = row['graph_path']
        if not os.path.exists(gpath):
            print(f"[warn] missing graph: {gpath} -> skipped")
            continue
        g = torch.load(gpath)
        x = g.get('node_attr')
        edge_index = g.get('edge_index')
        edge_attr = g.get('edge_attr', None)
        t1 = row.get(target_cols[0], np.nan)
        t2 = row.get(target_cols[1], np.nan)
        if pd.isna(t1) and replace_nan_with is not None: t1 = replace_nan_with
        if pd.isna(t2) and replace_nan_with is not None: t2 = replace_nan_with
        y = torch.tensor([float(t1) if not pd.isna(t1) else float('nan'),
                          float(t2) if not pd.isna(t2) else float('nan')], dtype=torch.float)
        data_list.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'smiles': g.get('smiles'), 'ligand_id': g.get('ligand_id'), 'graph_path': gpath})
    return data_list

# -------------------------
# Dataset Constructions
# -------------------------

class GraphDictDataset(Dataset):
    def __init__(self, dict_list):
        self.list = dict_list
    def __len__(self):
        return len(self.list)
    def __getitem__(self, idx):
        return self.list[idx]

def collate_graphs(graphs):
    xs, eis, eas, ys, node_counts, ligand_ids, smiles = [], [], [], [], [], [], []
    node_offset = 0
    for g in graphs:
        x = g['x']
        if x is None: x = torch.empty((0,0))
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float)
        xs.append(x)
        
        ei = g.get('edge_index')
        shifted_ei = (ei.clone() if isinstance(ei, torch.Tensor) and ei.numel() > 0 else torch.empty((2,0), dtype=torch.long)) + node_offset
        eis.append(shifted_ei)

        ea = g.get('edge_attr')
        if ea is not None and not isinstance(ea, torch.Tensor): ea = torch.tensor(ea, dtype=torch.float)
        eas.append(ea)
        
        ys.append(g.get('y', torch.tensor([float('nan'), float('nan')])).view(-1))
        num_nodes = x.shape[0] if hasattr(x, 'shape') else 0
        node_counts.append(num_nodes)
        ligand_ids.append(g.get('ligand_id'))
        smiles.append(g.get('smiles'))
        node_offset += num_nodes

    batch_x = torch.cat(xs, dim=0) if xs and any(t.numel() > 0 for t in xs) else torch.empty(0)
    batch_ei = torch.cat(eis, dim=1) if eis and any(t.numel() > 0 for t in eis) else torch.empty(2, 0, dtype=torch.long)
    batch_ea = torch.cat([e for e in eas if e is not None and e.numel() > 0], dim=0) if eas and any(e is not None and e.numel() > 0 for e in eas) else torch.empty(0)
    batch_y = torch.stack(ys, dim=0) if ys else torch.empty(0, 2)
    batch_vec = torch.repeat_interleave(torch.arange(len(node_counts)), torch.tensor(node_counts, dtype=torch.long))
    
    return {'x': batch_x, 'edge_index': batch_ei, 'edge_attr': batch_ea, 'y': batch_y, 'node_counts': node_counts, 'batch_vec': batch_vec, 'ligand_id': ligand_ids, 'smiles': smiles}

# -------------------------
# SiLU-GNN models
# - GINEconv
# - GATv2
# - Transformer
# - GCN
# -------------------------

class GNNMember(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, conv_type='gcn', use_pyg=USE_PYG):
        super().__init__()
        self.use_pyg = use_pyg and USE_PYG
        self.conv_type = conv_type.lower()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if self.use_pyg:
            self.convs = nn.ModuleList()
            
class GNNMember(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, conv_type='gcn', edge_dim=None, use_pyg=USE_PYG):
        super().__init__()
        self.use_pyg = use_pyg and USE_PYG
        self.conv_type = conv_type.lower()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if self.use_pyg:
            self.convs = nn.ModuleList()
            
            if 'gine' in self.conv_type:
                if edge_dim is None:
                    raise ValueError("edge_dim must be specified for GINEConv")
                    
                self.convs.append(GINEConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)), edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)), edge_dim=edge_dim))
            
            elif 'transformer' in self.conv_type:
                if edge_dim is None:
                    raise ValueError("edge_dim must be specified for TransformerConv")

                self.convs.append(TransformerConv(in_dim, hidden_dim, edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim))            
            
            else:
                ConvLayer = GCNConv
                if 'gatv2' in self.conv_type: ConvLayer = GATv2Conv
                elif 'transformer' in self.conv_type: ConvLayer = TransformerConv
                self.convs.append(ConvLayer(in_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(ConvLayer(hidden_dim, hidden_dim))
            
            self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            
        else:
            self.mlp_node = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        
        self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.head_vina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float, device=next(self.parameters()).device)
        
        if self.use_pyg:
            h = x
            for conv in self.convs:
                if self.conv_type in ['transformer', 'gine'] and edge_attr is not None and edge_attr.numel() > 0:
                    h = conv(h, edge_index, edge_attr)
                else:
                    h = conv(h, edge_index)
                    
                h = F.silu(h)
                h = self.dropout(h)
            
            hg = self.attention_pool(h, batch_vec) if batch_vec is not None and batch_vec.numel() > 0 else h.mean(dim=0, keepdim=True)
            
        else:
            h_nodes = self.mlp_node(x) if x.numel() > 0 else torch.zeros((0, self.hidden_dim), device=x.device)
            if batch_vec is None or batch_vec.numel() == 0:
                hg = h_nodes.mean(dim=0, keepdim=True)
            else:
                num_graphs = batch_vec.max().item() + 1
                hg = torch.zeros(num_graphs, h_nodes.size(1), device=h_nodes.device)
                hg.index_add_(0, batch_vec, h_nodes)
                node_counts = torch.bincount(batch_vec, minlength=num_graphs).unsqueeze(1).clamp(min=1)
                hg = hg / node_counts
            
        out1 = self.head_gnina(hg).view(-1)
        out2 = self.head_vina(hg).view(-1)
        return torch.stack([out1, out2], dim=1)

# ------------------------------------
# Concordance Correlation Coefficient
# ------------------------------------

class CCCLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, y):
        y_mean = torch.mean(y)
        preds_mean = torch.mean(preds)
        y_var = torch.var(y)
        preds_var = torch.var(preds)
        
        covariance = torch.mean((preds - preds_mean) * (y - y_mean))
        ccc = (2 * covariance) / (preds_var + y_var + (preds_mean - y_mean)**2 + self.epsilon)

        return 1 - ccc

# -------------------------
# Helpers
# -------------------------

def batch_to_tensors_for_model(batch, device):
    x = batch['x'].to(device)
    edge_index = batch['edge_index'].to(device)
    batch_vec = batch['batch_vec'].to(device)
    edge_attr = batch.get('edge_attr')
    edge_attr = edge_attr.to(device) if edge_attr is not None and edge_attr.numel() > 0 else None
    y = batch['y'].to(device)
    return x, edge_index, batch_vec, edge_attr, y

def train_one_epoch_log(model, optimizer, train_loader, device, criterion):
    model.train()
    batch_losses = []
    for batch in train_loader:
        x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
        optimizer.zero_grad()
        preds = model(x, edge_index, batch_vec, edge_attr)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    return batch_losses

def build_ensemble_from_specs(specs, in_dim, device, hidden_dim, num_layers, dropout, edge_dim):
    models = []
    for spec in specs:
        seed = spec.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        conv_type = spec.get('conv_type', 'gcn')
        m = GNNMember(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, conv_type=conv_type, edge_dim=edge_dim)
        m.to(device)
        models.append(m)
    return models
         
# -------------------------
# Monte-Carlo Sampling
# -------------------------

def enable_mc_dropout(model, enable=True):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(enable)

def sample_ensemble_predictions(ensemble_models, loader, device, mc_T=5, enable_mc=False, verbose=False):
    all_preds_draws = []
    batches = list(loader)
    N = sum(b['y'].shape[0] for b in batches)
    if N == 0: return np.zeros((0, 0, 0))

    for m_idx, m in enumerate(ensemble_models):
        conv_type = m.conv_type if hasattr(m, 'conv_type') else 'unknown'
        if verbose: print(f" Sampling from model {m_idx+1}/{len(ensemble_models)} ({conv_type.upper()}) (MC-T={mc_T if enable_mc else 1})")
        
        num_draws = mc_T if enable_mc and mc_T > 0 else 1
        for draw in range(num_draws):
            enable_mc_dropout(m, enable=enable_mc)
            preds_list = []
            with torch.no_grad():
                for batch in batches:
                    x, edge_index, batch_vec, edge_attr, _ = batch_to_tensors_for_model(batch, device)
                    out = m(x, edge_index, batch_vec, edge_attr)
                    preds_list.append(out.cpu().numpy())
            preds_cat = np.concatenate(preds_list, axis=0)
            all_preds_draws.append(preds_cat)

    enable_mc_dropout(m, False)
    if not all_preds_draws: return np.zeros((0, 0, 0))
    return np.stack(all_preds_draws, axis=0)

# -------------------------
# Affine Calibration
# -------------------------

def fit_affine_map(x, y, use_robust=False):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2: return 1.0, 0.0
    xs, ys = x[mask].reshape(-1, 1), y[mask]
    if use_robust:
        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor(random_state=42).fit(xs, ys)
            a, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            return float(a), float(b)
        except ImportError:
            print("[warn] scikit-learn not found for RANSAC. Falling back to OLS.")
    a, b = np.polyfit(xs.ravel(), ys, 1)
    return float(a), float(b)

# -------------------------
# Main
# -------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Using device: {device}")

    ## Data loading and spliting ##
    
    data_list = load_graphs_from_meta(args.meta_csv, graphs_root=args.graphs_root, replace_nan_with=args.replace_nan)
    if not data_list: raise RuntimeError("No data loaded.")
    print(f"[info] {len(data_list)} graphs loaded.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    idxs = np.arange(len(data_list)); np.random.shuffle(idxs)
    n = len(idxs)
    ntrain = int(n * args.train_frac); nval = int(n * args.val_frac)
    train_idx, val_idx, test_idx = idxs[:ntrain], idxs[ntrain:ntrain+nval], idxs[ntrain+nval:]
    train_list, val_list, test_list = [data_list[i] for i in train_idx], [data_list[i] for i in val_idx], [data_list[i] for i in test_idx]
    print(f"[info] Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    train_loader = DataLoader(GraphDictDataset(train_list), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(GraphDictDataset(val_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(GraphDictDataset(test_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    in_dim = next((d['x'].shape[1] for d in data_list if d['x'] is not None and d['x'].ndim == 2 and d['x'].shape[1] > 0), args.fallback_in_dim)
    edge_dim = next((d['edge_attr'].shape[1] for d in data_list if d.get('edge_attr') is not None and d['edge_attr'].ndim == 2 and d['edge_attr'].shape[1] > 0), None)
    print(f"[info] Determined input feature dimension (in_dim): {in_dim}")
    print(f"[info] Determined edge feature dimension (edge_dim): {edge_dim}")

    y_train = np.array([d['y'].numpy() for d in train_list])
    mean_train = np.nanmean(y_train, axis=0, keepdims=True)
    std_train = np.nanstd(y_train, axis=0, keepdims=True)
    std_train = np.where(std_train == 0, 1.0, std_train)
    print(f"[info] Train target mean: {mean_train.ravel()}, std: {std_train.ravel()}")

    ccc_loss_fn = CCCLoss()
    def criterion(preds, y):
        y_norm = (y - torch.tensor(mean_train, dtype=torch.float, device=preds.device)) / torch.tensor(std_train, dtype=torch.float, device=preds.device)
        mask_gnina = ~torch.isnan(y_norm[:, 0])
        if mask_gnina.sum() > 1:
            loss_gnina = ccc_loss_fn(preds[:, 0][mask_gnina], y_norm[:, 0][mask_gnina])
        else:
            loss_gnina = torch.tensor(0.0, device=preds.device)
        mask_vina = ~torch.isnan(y_norm[:, 1])
        if mask_vina.sum() > 1:
            loss_vina = ccc_loss_fn(preds[:, 1][mask_vina], y_norm[:, 1][mask_vina])
        else:
            loss_vina = torch.tensor(0.0, device=preds.device)
        total_loss = loss_gnina + loss_vina
        return total_loss
        
    ensemble_specs = json.loads(args.ensemble_specs)
    ensemble_models = build_ensemble_from_specs(ensemble_specs, in_dim, device, args.hidden_dim, args.num_layers, args.dropout, edge_dim=edge_dim)
    print(f"[info] Built ensemble with {len(ensemble_models)} members.")

    ## Ensemble learning ##
    
    all_epoch_logs = []

    for i, model in enumerate(ensemble_models):
        spec = ensemble_specs[i]
        model_name = spec.get('name', 'model')
        print(f"\n--- Training ensemble member {i+1}/{len(ensemble_models)}: {model_name.upper()} ---")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = out_dir / f"ensemble_member_{i}_{model_name}_best.pt"

        for ep in range(1, args.epochs + 1):
            model.train()
            train_losses = train_one_epoch_log(model, optimizer, train_loader, device, criterion)
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
                    preds = model(x, edge_index, batch_vec, edge_attr)
                    val_losses.append(criterion(preds, y).item())
            
            mean_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            print(f"[Train] Ep {ep}/{args.epochs}, Mean Train Loss: {mean_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            all_epoch_logs.append({
                'model_name': model_name,
                'epoch': ep,
                'train_loss': mean_train_loss,
                'val_loss': avg_val_loss
            })
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if args.save_checkpoints:
                    torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {ep} due to no improvement in validation loss.")
                break
        
        if args.save_checkpoints and best_model_path.exists():
            print(f"Loading best model weights from: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))

    log_df = pd.DataFrame(all_epoch_logs)
    log_csv_path = out_dir / "epoch_training_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    print(f"\n[info] Saved epoch-by-epoch training and validation logs to: {log_csv_path}")

    ## Ensemble sampling ##
    
    S_samples = sample_ensemble_predictions(ensemble_models, test_loader, device, args.mc_T, args.enable_mc_dropout, verbose=True)
    if S_samples.size == 0: raise RuntimeError("Sampling returned no predictions.")
    S, N, C = S_samples.shape
    
    ## Rescaling using min-max / affine ##
    
    samples_orig = S_samples * std_train + mean_train
    mean_preds_orig = samples_orig.mean(axis=0)
    true_gnina = np.array([d['y'][0].item() if d['y'].numel() >= 1 else np.nan for d in test_list])
    true_vina = np.array([d['y'][1].item() if d['y'].numel() >= 2 else np.nan for d in test_list])
    pred_g, pred_v = mean_preds_orig[:, 0], mean_preds_orig[:, 1]

    pred_v_cal, a_v, b_v = pred_v, 1.0, 0.0
        
    def minmax_rescale(pred, target):
        mask = ~np.isnan(target)
        if mask.sum() == 0: return pred
        tmin, tmax = np.nanmin(target), np.nanmax(target)
        pmin, pmax = np.nanmin(pred), np.nanmax(pred)
        if np.isclose(pmax, pmin): return np.full_like(pred, np.nanmean(target))
        return (pred - pmin) / (pmax - pmin) * (tmax - tmin) + tmin

    if args.rescale == 'minmax':
        pred_g_rescaled = minmax_rescale(pred_g, true_gnina)
        pred_v_rescaled = minmax_rescale(pred_v, true_vina) 
        print("[info] Applied min-max rescaling to predictions.")
    else:
        pred_g_rescaled = pred_g
        pred_v_rescaled = pred_v

    if args.calibrate_affine:
        a_g, b_g = fit_affine_map(pred_g_rescaled, true_gnina, args.use_robust_calib)
        pred_g_cal = a_g * pred_g_rescaled + b_g
        a_v, b_v = fit_affine_map(pred_v_rescaled, true_vina, args.use_robust_calib)
        pred_v_cal = a_v * pred_v_rescaled + b_v
    else:
        pred_g_cal, a_g, b_g = pred_g_rescaled, 1.0, 0.0
        pred_v_cal, a_v, b_v = pred_v_rescaled, 1.0, 0.0

    ## Rank Confidence ##
    
    std_per_item_g = np.nanstd(samples_orig[:, :, 0], axis=0)
    score = pred_g_cal - args.uncertainty_lambda * std_per_item_g if args.uncertainty_lambda > 0 else pred_g_cal
    ranks_s = np.argsort(-samples_orig[:, :, 0], axis=1).argsort(axis=1)
    mean_rank, std_rank = ranks_s.mean(axis=0), ranks_s.std(axis=0)
    rank_confidence = 1.0 - (mean_rank / (max(1, N - 1)))

    mask_g = ~np.isnan(true_gnina) & ~np.isnan(pred_g_cal)
    if mask_g.sum() > 1:
        corr_g = pearsonr(true_gnina[mask_g], pred_g_cal[mask_g])[0]
        print(f"\n[Performance] Final GNINA Pearson Correlation: {corr_g:.4f}")

    mask_v = ~np.isnan(true_vina) & ~np.isnan(pred_v_cal)
    if mask_v.sum() > 1:
        corr_v = pearsonr(true_vina[mask_v], pred_v_cal[mask_v])[0]
        print(f"[Performance] Final VINA  Pearson Correlation: {corr_v:.4f}\n")
    
    df_out = pd.DataFrame({
        'ligand_id': [d.get('ligand_id') for d in test_list],
        'smiles': [d.get('smiles') for d in test_list],
        'true_gnina': true_gnina, 'true_vina': true_vina,
        'mean_gnina': pred_g, 'std_gnina': std_per_item_g,
        'mean_gnina_calibrated': pred_g_cal,
        'final_score': score,
        'mean_rank': mean_rank, 'std_rank': std_rank, 'rank_confidence': rank_confidence,
        'calib_a_g': a_g, 'calib_b_g': b_g
    })
    df_sorted = df_out.sort_values('final_score', ascending=False).reset_index(drop=True)
    out_csv = out_dir / "df_sorted_final.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n[done] Saved final sorted results to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgraded Bayesian GNN Ensemble Pipeline")
    
    ## paths
    parser.add_argument('--meta_csv', type=str, default='/home/ssm-user/project/scores/processed_graphs.csv', help="Path to the metadata CSV file.")
    parser.add_argument('--graphs_root', type=str, default='/home/ssm-user/project/graphs', help="Directory containing the graph .pt files.")
    parser.add_argument('--out_dir', type=str, default='/home/ssm-user/project/scores/results', help="Directory to save results.")
    
    ## learning parameters
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs for each ensemble member.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_corr', type=float, default=1.0, help="Weight for the Pearson correlation loss term.")
    parser.add_argument('--early_stopping_patience', type=int, default=25, help="Patience for early stopping.")

    ## model construction
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension size.")
    parser.add_argument('--num_layers', type=int, default=5, help="Number of GNN layers.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")
    parser.add_argument('--fallback_in_dim', type=int, default=52)
    
    ## data configure
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--replace_nan', type=float, default=None)
    
    ## ensemble and sampling
    parser.add_argument('--ensemble_specs', type=str,
                        default='[{"name":"gine","conv_type":"gine","seed":42},{"name":"gatv2","conv_type":"gatv2","seed":43},{"name":"transformer","conv_type":"transformer","seed":44},{"name":"gcn","conv_type":"gcn","seed":45}]',
                        help="JSON string defining the ensemble members.")
    parser.add_argument('--mc_T', type=int, default=5)
    parser.add_argument('--enable_mc_dropout', action='store_true')
    
    ## callibrations
    parser.add_argument('--rescale', type=str, choices=['none','minmax'], default='minmax')
    parser.add_argument('--calibrate_affine', action='store_true', default=True)
    parser.add_argument('--use_robust_calib', action='store_true')
    parser.add_argument('--uncertainty_lambda', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_checkpoints', action='store_true', help="Save best model checkpoints during training.")
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    main(args)
