#!/usr/bin/env python3
"""
09_bayesian_gnn_calibrated.py

Pipeline:
 - load graph .pt via metadata CSV (graph_path, smiles, ligand_id, gnina_affinity, vina_affinity)
 - train a small GNN (warmup) if requested (minimal; can be disabled), build ensemble from specs
 - sample ensemble with MC-dropout to produce S_samples (S x N x 2)
 - denormalize, min-max rescale to true range (optional)
 - affine calibration pred -> true (OLS or RANSAC if sklearn available)
 - optional uncertainty-penalty re-ranking (score = pred_rescaled - lambda*std)
 - save df_sorted_final.csv with required columns
"""
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
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

# Try import torch_geometric convs (optional)
USE_PYG = False
try:
    from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, global_mean_pool
    USE_PYG = True
except Exception:
    USE_PYG = False

# -------------------------
# utilities: load graphs metadata and graph .pt objects
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
        # targets
        t1 = row.get(target_cols[0], np.nan)
        t2 = row.get(target_cols[1], np.nan)
        if pd.isna(t1) and replace_nan_with is not None:
            t1 = replace_nan_with
        if pd.isna(t2) and replace_nan_with is not None:
            t2 = replace_nan_with
        y = torch.tensor([float(t1) if (not pd.isna(t1)) else float('nan'),
                          float(t2) if (not pd.isna(t2)) else float('nan')], dtype=torch.float)
        data_list.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'smiles': g.get('smiles'), 'ligand_id': g.get('ligand_id'), 'graph_path': gpath})
    return data_list

# -------------------------
# Dataset & collate
# -------------------------
class GraphDictDataset(Dataset):
    def __init__(self, dict_list):
        self.list = dict_list
    def __len__(self):
        return len(self.list)
    def __getitem__(self, idx):
        return self.list[idx]

def collate_graphs(graphs):
    xs = []
    eis = []
    eas = []
    ys = []
    node_counts = []
    ligand_ids = []
    smiles = []
    node_offset = 0
    for g in graphs:
        x = g['x']
        if x is None:
            x = torch.empty((0,0))
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        xs.append(x)
        ei = g.get('edge_index')
        if ei is None or (isinstance(ei, torch.Tensor) and ei.numel()==0):
            shifted_ei = torch.empty((2,0), dtype=torch.long)
        else:
            shifted_ei = ei.clone() if isinstance(ei, torch.Tensor) else torch.tensor(ei, dtype=torch.long).t().contiguous()
            shifted_ei = shifted_ei + node_offset
        eis.append(shifted_ei)
        ea = g.get('edge_attr')
        if ea is None:
            eas.append(torch.empty((0,)))
        else:
            if not isinstance(ea, torch.Tensor):
                ea = torch.tensor(ea, dtype=torch.float)
            eas.append(ea)
        y = g.get('y', torch.tensor([float('nan'), float('nan')]))
        ys.append(y.view(-1))
        node_counts.append(x.shape[0] if hasattr(x,'shape') else 0)
        ligand_ids.append(g.get('ligand_id'))
        smiles.append(g.get('smiles'))
        node_offset += x.shape[0] if hasattr(x,'shape') else 0

    # concat x
    if len(xs)>0 and any([xx.numel()>0 for xx in xs]):
        xs2=[]
        for xx in xs:
            if xx.numel()==0:
                xs2.append(torch.empty((0, xs[0].shape[1] if xs[0].ndim>1 else 0)))
            else:
                xs2.append(xx)
        batch_x = torch.cat(xs2, dim=0)
    else:
        batch_x = torch.empty((0,))

    if len(eis)>0 and any([e.numel()>0 for e in eis]):
        batch_ei = torch.cat([e if e.numel()>0 else torch.empty((2,0), dtype=torch.long) for e in eis], dim=1)
    else:
        batch_ei = torch.empty((2,0), dtype=torch.long)

    if len(eas)>0 and any([e.numel()>0 for e in eas]):
        # try to handle edge_attr shape differences
        try:
            batch_ea = torch.cat([e if e.numel()>0 else torch.empty((0, eas[0].shape[1] if hasattr(eas[0],'ndim') and eas[0].ndim>1 else 0)) for e in eas], dim=0)
        except Exception:
            batch_ea = torch.empty((0,))
    else:
        batch_ea = torch.empty((0,))

    batch_y = torch.stack(ys, dim=0) if len(ys)>0 else torch.empty((0,2))
    if len(node_counts)==0 or sum(node_counts)==0:
        batch_vec = torch.tensor([], dtype=torch.long)
    else:
        batch_vec = torch.repeat_interleave(torch.arange(len(node_counts), dtype=torch.long), torch.tensor(node_counts, dtype=torch.long))
    return {'x': batch_x, 'edge_index': batch_ei, 'edge_attr': batch_ea, 'y': batch_y, 'node_counts': node_counts, 'batch_vec': batch_vec, 'ligand_id': ligand_ids, 'smiles': smiles}

# -------------------------
# Model definition (GNNMember supporting conv_type)
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
            # choose conv layer class per conv_type
            Conv = GCNConv
            if 'gat' in self.conv_type:
                Conv = GATv2Conv if 'gatv2' in self.conv_type and 'GATv2Conv' in globals() else None
            if 'transformer' in self.conv_type:
                Conv = TransformerConv if 'TransformerConv' in globals() else None
            # fallback to GCNConv if specific not available
            if Conv is None:
                Conv = GCNConv
            self.convs.append(Conv(in_dim, hidden_dim))
            for _ in range(num_layers-1):
                self.convs.append(Conv(hidden_dim, hidden_dim))
            self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,1))
            self.head_vina  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,1))
        else:
            # simple MLP per node + mean pooling
            self.mlp_node = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,1))
            self.head_vina  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,1))

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=next(self.parameters()).device)
        if self.use_pyg:
            h = x
            for conv in self.convs:
                h = conv(h, edge_index)
                h = torch.relu(h)
                h = self.dropout(h)
            if batch_vec is None or batch_vec.numel()==0:
                hg = h.mean(dim=0, keepdim=True)
            else:
                hg = global_mean_pool(h, batch_vec)
        else:
            h_nodes = self.mlp_node(x) if x.numel()>0 else torch.zeros((batch_vec.shape[0] if batch_vec is not None else 1, self.hidden_dim), device=x.device)
            if batch_vec is None or batch_vec.numel()==0:
                hg = h_nodes.mean(dim=0, keepdim=True)
            else:
                bsize = int(batch_vec.max().item())+1 if batch_vec.numel()>0 else 1
                sums = torch.zeros((bsize, h_nodes.shape[1]), device=h_nodes.device)
                counts = torch.zeros((bsize,), device=h_nodes.device)
                for i in range(h_nodes.shape[0]):
                    bi = int(batch_vec[i].item())
                    sums[bi] += h_nodes[i]
                    counts[bi] += 1.0
                counts = counts.unsqueeze(1).clamp(min=1.0)
                hg = sums / counts
        out1 = self.head_gnina(hg).view(-1)
        out2 = self.head_vina(hg).view(-1)
        out = torch.stack([out1, out2], dim=1)
        return out

# -------------------------
# helpers: batch -> tensors
# -------------------------
def batch_to_tensors_for_model(batch, device):
    if isinstance(batch, dict) and 'x' in batch and isinstance(batch['x'], torch.Tensor):
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device) if isinstance(batch['edge_index'], torch.Tensor) else None
        batch_vec = batch['batch_vec'].to(device) if isinstance(batch.get('batch_vec', None), torch.Tensor) else None
        edge_attr = batch['edge_attr'].to(device) if isinstance(batch.get('edge_attr', None), torch.Tensor) else None
        y = batch['y'].to(device) if isinstance(batch.get('y', None), torch.Tensor) else torch.tensor([], device=device)
    else:
        raise RuntimeError("Unexpected batch format.")
    return x, edge_index, batch_vec, edge_attr, y

def train_one_epoch_log(model, optimizer, train_loader, device, criterion, kl_weight=0.0):
    model.train()
    batch_losses=[]
    for batch in train_loader:
        x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
        optimizer.zero_grad()
        preds = model(x, edge_index, batch_vec, edge_attr)
        loss = criterion(preds, y)
        if hasattr(model, 'kl_divergence') and callable(getattr(model, 'kl_divergence')) and kl_weight>0:
            try:
                loss = loss + kl_weight * model.kl_divergence()
            except Exception:
                pass
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.detach().cpu().item()))
    return batch_losses

# -------------------------
# build ensemble helper
# -------------------------
def build_ensemble_from_specs(specs, in_dim, device, hidden_dim=128, num_layers=3, dropout=0.2):
    models=[]
    for spec in specs:
        seed = spec.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        conv_type = spec.get('conv_type', spec.get('name', 'gcn'))
        try:
            m = GNNMember(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, conv_type=conv_type)
        except Exception:
            m = GNNMember(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        m.to(device)
        m.eval()
        models.append(m)
    return models

# -------------------------
# sampling: ensemble + MC
# -------------------------
def enable_mc_dropout_on_model(model, enable=True):
    # set dropout modules train/eval accordingly
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            if enable:
                m.train()
            else:
                m.eval()

def sample_ensemble_predictions(ensemble_models, loader, device, mc_T=5, enable_mc=False, verbose=False):
    """
    Return S_samples array shape (S, N, 2) where S = len(ensemble_models) * mc_T (if mc_T>0)
    Sampling order follows ensemble members then draws: for m in models: for t in mc_T: collect predictions over loader (in order)
    """
    if not isinstance(ensemble_models, (list, tuple)) or len(ensemble_models)==0:
        raise RuntimeError("No ensemble models provided.")
    # first compute number of test items N by iterating once
    all_preds_draws = []
    N = 0
    # create list of batches (materialize) to ensure consistent order and avoid re-iter issues
    batches = list(loader)
    # compute N by summing batch_vec sizes or y rows
    for b in batches:
        if isinstance(b, dict) and 'y' in b:
            N += b['y'].shape[0]
        else:
            # try attribute
            y = getattr(b, 'y', None)
            if isinstance(y, torch.Tensor):
                N += y.shape[0]
    if N == 0:
        return np.zeros((0,0,0))
    # iterate models
    for m_idx, m in enumerate(ensemble_models):
        if verbose:
            print(f"  model {m_idx+1}/{len(ensemble_models)} sampling (mc={mc_T}, enable_mc={enable_mc})")
        for draw in range(mc_T if mc_T>0 else 1):
            # if enabling mc dropout, set dropout train mode for this draw
            if enable_mc:
                enable_mc_dropout_on_model(m, True)
            else:
                enable_mc_dropout_on_model(m, False)
            preds_list = []
            with torch.no_grad():
                for batch in batches:
                    x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
                    out = m(x, edge_index, batch_vec, edge_attr)
                    preds_list.append(out.detach().cpu().numpy())
            # concat
            preds_cat = np.concatenate(preds_list, axis=0) if len(preds_list)>0 else np.zeros((0,2))
            if preds_cat.shape[0] != N:
                # mismatch: try to reshuffle by y length per batch; but we assume concatenation aligned
                if verbose:
                    print(f"[warn] prediction length {preds_cat.shape[0]} != N {N}")
                # pad or truncate
                if preds_cat.shape[0] < N:
                    pad = np.tile(preds_cat[-1:], (N - preds_cat.shape[0], 1))
                    preds_cat = np.concatenate([preds_cat, pad], axis=0)
                else:
                    preds_cat = preds_cat[:N]
            all_preds_draws.append(preds_cat)
            if verbose:
                print(f"    draw {draw+1}/{mc_T} collected, shape {preds_cat.shape}")
            # after draw, put dropout back to eval (we'll control next draw)
            enable_mc_dropout_on_model(m, False)
    S = len(all_preds_draws)
    if S == 0:
        return np.zeros((0,0,0))
    samples = np.stack(all_preds_draws, axis=0)  # (S, N, 2)
    return samples

# -------------------------
# affine calibration helper
# -------------------------
def fit_affine_map(x, y, use_robust=False):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return 1.0, 0.0
    xs = x[mask].reshape(-1,1)
    ys = y[mask].reshape(-1,1)
    if use_robust:
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            base = LinearRegression()
            ransac = RANSACRegressor(base_estimator=base, random_state=42)
            ransac.fit(xs, ys.ravel())
            # estimator_ is fitted LinearRegression
            est = ransac.estimator_
            a = float(est.coef_[0])
            b = float(est.intercept_)
            return a, b
        except Exception:
            pass
    # fallback OLS
    a, b = np.polyfit(xs.ravel(), ys.ravel(), 1)
    return float(a), float(b)

# -------------------------
# main
# -------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading meta:", args.meta_csv)
    data_list = load_graphs_from_meta(args.meta_csv, graphs_root=args.graphs_root, target_cols=('gnina_affinity','vina_affinity'), replace_nan_with=args.replace_nan)
    if len(data_list) == 0:
        raise RuntimeError("No data loaded.")
    print(f"[info] {len(data_list)} graphs loaded.")

    # split
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    idxs = np.arange(len(data_list)); np.random.shuffle(idxs)
    n = len(idxs)
    ntrain = int(n * args.train_frac); nval = int(n * args.val_frac)
    train_idx = idxs[:ntrain]; val_idx = idxs[ntrain:ntrain+nval]; test_idx = idxs[ntrain+nval:]
    train_list = [data_list[i] for i in train_idx]; val_list = [data_list[i] for i in val_idx]; test_list = [data_list[i] for i in test_idx]
    print(f"[info] split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    # dataloaders
    train_loader = DataLoader(GraphDictDataset(train_list), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader   = DataLoader(GraphDictDataset(val_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(GraphDictDataset(test_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    # determine in_dim
    in_dim = None
    for d in train_list + val_list + test_list:
        if d['x'] is not None and isinstance(d['x'], torch.Tensor) and d['x'].ndim==2 and d['x'].shape[1]>0:
            in_dim = d['x'].shape[1]; break
    if in_dim is None:
        in_dim = args.fallback_in_dim
        print("[warn] using fallback in_dim", in_dim)
    print("[info] in_dim =", in_dim)

    # compute train target mean/std
    y_train = np.stack([d['y'].numpy() for d in train_list], axis=0) if len(train_list)>0 else np.zeros((0,2))
    mean_train = np.nanmean(y_train, axis=0) if y_train.size>0 else np.array([0.0, 0.0])
    std_train = np.nanstd(y_train, axis=0) if y_train.size>0 else np.array([1.0, 1.0])
    std_train = np.where(std_train==0, 1.0, std_train)
    # replace nans if needed
    for i in range(mean_train.shape[0]):
        if np.isnan(mean_train[i]):
            if args.replace_nan is not None:
                mean_train[i] = args.replace_nan
                std_train[i] = 1.0
            else:
                mean_train[i] = 0.0; std_train[i] = 1.0
    print("[info] train target mean/std:", mean_train, std_train)

    # optional warm-up single model training (minimal). We will still build ensemble later.
    if args.do_train:
        print("[info] training warmup single model...")
        single_model = GNNMember(in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, conv_type='gcn')
        single_model.to(device)
        optimizer = torch.optim.Adam(single_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        def criterion(preds, y):
            y_np = y.detach().cpu().numpy()
            mask = ~np.isnan(y_np)
            if not mask.any():
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
            y_norm = (np.where(mask, y_np, mean_train.reshape(1,2)) - mean_train.reshape(1,2)) / std_train.reshape(1,2)
            y_norm_t = torch.tensor(y_norm, dtype=torch.float, device=preds.device)
            se = (preds - y_norm_t)**2
            mask_t = torch.tensor(mask.astype(float), dtype=torch.float, device=preds.device)
            loss = (se * mask_t).sum() / (mask_t.sum() + 1e-12)
            return loss
        for ep in range(1, args.epochs+1):
            batch_losses = train_one_epoch_log(single_model, optimizer, train_loader, device, criterion, kl_weight=args.kl_weight)
            print(f"[train] ep {ep}/{args.epochs} mean_loss {np.mean(batch_losses):.6f}")
        # optionally save warmup model
        if args.save_checkpoints:
            torch.save(single_model.state_dict(), out_dir / "warmup_model.pt")
    else:
        single_model = None

    # build ensemble
    print("[info] building ensemble from specs")
    try:
        ensemble_specs = args.ensemble_specs
        if isinstance(ensemble_specs, str):
            ensemble_specs = json.loads(ensemble_specs)
    except Exception:
        ensemble_specs = [
            {'name':'gcn','conv_type':'gcn','seed':42},
            {'name':'gatv2','conv_type':'gatv2','seed':43},
            {'name':'transformer','conv_type':'transformer','seed':44}
        ]
    ensemble_models = build_ensemble_from_specs(ensemble_specs, in_dim, device, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)

    # optionally initialize ensemble from warmup
    if args.init_ensemble_from_trained and single_model is not None:
        s_state = single_model.state_dict()
        for em in ensemble_models:
            try:
                em.load_state_dict(s_state, strict=False)
            except Exception:
                pass

    # sampling
    print("[info] sampling ensemble (mc_T={}, enable_mc={})".format(args.mc_T, args.enable_mc_dropout))
    S_samples = sample_ensemble_predictions(ensemble_models, test_loader, device, mc_T=args.mc_T, enable_mc=args.enable_mc_dropout, verbose=True)
    if S_samples.size == 0:
        raise RuntimeError("Empty samples.")
    S, N, C = S_samples.shape
    print(f"[info] sampled S={S}, N={N}, C={C}")

    # denormalize to original scale using mean_train/std_train
    samples_orig = S_samples * std_train.reshape(1,1,2) + mean_train.reshape(1,1,2)   # (S,N,2)
    mean_preds = samples_orig.mean(axis=0)  # (N,2)
    mean_preds_orig = mean_preds.copy()

    # build true arrays in test order
    true_gnina = np.array([ (float(d['y'][0].item()) if isinstance(d['y'], torch.Tensor) and d['y'].numel()>=1 and not torch.isnan(d['y'][0]) else float('nan')) for d in test_list ])
    true_vina  = np.array([ (float(d['y'][1].item()) if isinstance(d['y'], torch.Tensor) and d['y'].numel()>=2 and not torch.isnan(d['y'][1]) else float('nan')) for d in test_list ])

    pred_g = mean_preds_orig[:,0].copy()
    pred_v = mean_preds_orig[:,1].copy()

    # optional rescale to true distribution (minmax)
    def minmax_rescale(pred, target):
        mask = ~np.isnan(target)
        if mask.sum() == 0:
            return pred.copy()
        tmin = float(np.nanmin(target)); tmax = float(np.nanmax(target))
        pmin = float(np.nanmin(pred)); pmax = float(np.nanmax(pred))
        if np.isclose(pmax, pmin):
            return np.full_like(pred, float(np.nanmean(target)))
        return (pred - pmin) / (pmax - pmin) * (tmax - tmin) + tmin

    if args.rescale == 'minmax':
        pred_g_rescaled = minmax_rescale(pred_g, true_gnina)
        pred_v_rescaled = minmax_rescale(pred_v, true_vina)
    else:
        pred_g_rescaled = pred_g.copy()
        pred_v_rescaled = pred_v.copy()

    # per-item std (gnina/vina)
    std_per_item_g = np.nanstd(samples_orig[:,:,0], axis=0)
    std_per_item_v = np.nanstd(samples_orig[:,:,1], axis=0)

    # pearson before/after rescale (print)
    try:
        mg = ~np.isnan(true_gnina)
        if mg.sum() > 1:
            print("[info] GNINA pearson before rescale:", pearsonr(true_gnina[mg], pred_g[mg])[0], "after rescale:", pearsonr(true_gnina[mg], pred_g_rescaled[mg])[0])
        mv = ~np.isnan(true_vina)
        if mv.sum() > 1:
            print("[info] VINA  pearson before rescale:", pearsonr(true_vina[mv], pred_v[mv])[0], "after rescale:", pearsonr(true_vina[mv], pred_v_rescaled[mv])[0])
    except Exception as e:
        print("[warn] pearson failed:", e)

    # AFFINE calibration (if requested)
    if args.calibrate_affine:
        a_g, b_g = fit_affine_map(pred_g_rescaled, true_gnina, use_robust=args.use_robust_calib)
        pred_g_cal = a_g * pred_g_rescaled + b_g
        a_v, b_v = fit_affine_map(pred_v_rescaled, true_vina, use_robust=args.use_robust_calib)
        pred_v_cal = a_v * pred_v_rescaled + b_v
        # show diagnostics
        try:
            maskg = ~np.isnan(true_gnina)
            if maskg.sum()>1:
                print(f"[calib] GNINA affine a={a_g:.6f}, b={b_g:.6f}, pearson before={pearsonr(true_gnina[maskg], pred_g_rescaled[maskg])[0]:.4f}, after={pearsonr(true_gnina[maskg], pred_g_cal[maskg])[0]:.4f}")
            maskv = ~np.isnan(true_vina)
            if maskv.sum()>1:
                print(f"[calib] VINA  affine a={a_v:.6f}, b={b_v:.6f}, pearson before={pearsonr(true_vina[maskv], pred_v_rescaled[maskv])[0]:.4f}, after={pearsonr(true_vina[maskv], pred_v_cal[maskv])[0]:.4f}")
        except Exception:
            pass
    else:
        pred_g_cal = pred_g_rescaled.copy()
        pred_v_cal = pred_v_rescaled.copy()
        a_g, b_g, a_v, b_v = 1.0, 0.0, 1.0, 0.0

    # final predictions used for ranking
    if args.uncertainty_lambda is not None and args.uncertainty_lambda > 0.0:
        lam = float(args.uncertainty_lambda)
        score = pred_g_cal - lam * std_per_item_g
        print(f"[info] using uncertainty-penalty lambda={lam} for ranking (score = pred - lambda*std)")
    else:
        score = None

    final_rank_vals = score if score is not None else pred_g_cal

    # compute S-based rank stats (based on gnina samples_orig)
    samples_g = samples_orig[:,:,0]  # (S,N)
    ranks_each_draw = np.argsort(-samples_g, axis=1)
    pos_each_draw = np.empty_like(ranks_each_draw)
    for i in range(ranks_each_draw.shape[0]):
        pos = np.empty(N, dtype=int)
        pos[ranks_each_draw[i]] = np.arange(N)
        pos_each_draw[i] = pos
    mean_rank = pos_each_draw.mean(axis=0)
    std_rank = pos_each_draw.std(axis=0)
    rank_confidence = 1.0 - (mean_rank / (max(1, N-1)))

    ligand_ids = [d.get('ligand_id') for d in test_list]
    smiles_list = [d.get('smiles') for d in test_list]

    df_out = pd.DataFrame({
        'ligand_id': ligand_ids,
        'smiles': smiles_list,
        'true_gnina': true_gnina,
        'true_vina': true_vina,
        'mean_gnina': pred_g,
        'mean_vina': pred_v,
        'mean_gnina_rescaled': pred_g_rescaled,
        'mean_vina_rescaled': pred_v_rescaled,
        'mean_gnina_calibrated': pred_g_cal,
        'mean_vina_calibrated': pred_v_cal,
        'std_gnina': std_per_item_g,
        'std_vina': std_per_item_v,
        'mean_rank': mean_rank,
        'std_rank': std_rank,
        'rank_confidence': rank_confidence
    })

    if score is not None:
        df_out['score'] = score

    # add calib params for info
    df_out['calib_a_g'] = a_g; df_out['calib_b_g'] = b_g
    df_out['calib_a_v'] = a_v; df_out['calib_b_v'] = b_v

    # sort by chosen ordering
    if 'score' in df_out.columns:
        df_sorted = df_out.sort_values('score', ascending=False).reset_index(drop=True)
    else:
        df_sorted = df_out.sort_values('mean_gnina_calibrated', ascending=False).reset_index(drop=True)

    out_csv = Path(args.out_dir) / "df_sorted_final.csv"
    df_sorted.to_csv(out_csv, index=False)
    print("[done] saved final csv:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian GNN ensemble with minmax rescale + affine calibration")
    parser.add_argument('--meta_csv', type=str, required=True)
    parser.add_argument('--graphs_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./results")
    parser.add_argument('--do_train', action='store_true', help="perform warmup single-model training")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--fallback_in_dim', type=int, default=52)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--replace_nan', type=float, default=None)
    parser.add_argument('--kl_weight', type=float, default=0.0)
    parser.add_argument('--mc_T', type=int, default=5)
    parser.add_argument('--enable_mc_dropout', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--ensemble_specs', type=str, default='[{"name":"gcn","conv_type":"gcn","seed":42},{"name":"gatv2","conv_type":"gatv2","seed":43},{"name":"transformer","conv_type":"transformer","seed":44}]')
    parser.add_argument('--init_ensemble_from_trained', action='store_true')
    parser.add_argument('--rescale', type=str, choices=['none','minmax'], default='minmax')
    parser.add_argument('--uncertainty_lambda', type=float, default=0.0)
    parser.add_argument('--calibrate_affine', action='store_true', default=True, help="apply affine calibration (pred->true)")
    parser.add_argument('--use_robust_calib', action='store_true', default=False, help="use RANSAC if sklearn available")
    args = parser.parse_args()
    main(args)
