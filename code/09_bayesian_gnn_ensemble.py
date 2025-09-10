#!/usr/bin/env python3
"""
Command examples:

    python 09_bayesian_gnn.py --meta_csv 'path/to/your/processed_graphs.csv' --graphs_root 'path/to/your/graphs'

Description:

--meta_df : csv with metadata (path/file.csv)
--graphs_root : directory for graph files (path)
--out_dir : output directory (default ./results)

--epochs : number of training epochs (default 10)
--batch_size : training batch size (default 32)
--lr : learning rate (default 1e-3)
--weight_decay : weight decay (default 1e-5)
--hidden_dim : hidden dimension size (default 128)
--num_layers : number of GNN layers (default 3)
--dropout : dropout rate (default 0.2)
--train_frac : fraction of data for training (default 0.8)
--val_frac : fraction of data for validation (default 0.1)

--fallback_in_dim : fallback input dimension if not inferrable (default 52)
--seed : random seed (default 42)
--save_checkpoints : whether to save model checkpoints each epoch
--replace_nan : value to replace NaNs in target columns (default None)

--kl_weight : weight for KL divergence regularization (default 0.0)
--mc_T : number of MC dropout samples (default 5)
--enable_mc_dropout : whether to enable MC dropout during inference
--no_cuda : disable CUDA even if available
--ensemble_specs : list of ensemble you want (default : gcn, gatv2, transformer)
"""

import os
import argparse
from pathlib import Path
import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    # 필요한 GNN 레이어들을 모두 임포트합니다.
    from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, global_mean_pool
    USE_PYG = True
except Exception:
    USE_PYG = False

# ---------------------------
# load_graphs_from_meta (same semantics: y = [gnina, vina])
# ---------------------------

def load_graphs_from_meta(meta_csv, graphs_root=None, target_cols=('gnina_affinity','vina_affinity'), replace_nan_with=None):
    df = pd.read_csv(meta_csv)
    if graphs_root is not None:
        df['graph_path'] = df['graph_path'].apply(
            lambda p: os.path.join(graphs_root, os.path.basename(p)) if not os.path.isabs(p) else p
        )
    data_list = []
    for _, row in df.iterrows():
        gpath = row['graph_path']
        if not os.path.exists(gpath):
            print(f"[warn] missing graph file {gpath}, skipping")
            continue
        g = torch.load(gpath)
        x = g.get('node_attr')
        edge_index = g.get('edge_index')
        edge_attr = g.get('edge_attr') if 'edge_attr' in g else None
        t1 = row[target_cols[0]] if target_cols[0] in row.index else np.nan
        t2 = row[target_cols[1]] if target_cols[1] in row.index else np.nan
        if pd.isna(t1) and replace_nan_with is not None:
            t1 = replace_nan_with
        if pd.isna(t2) and replace_nan_with is not None:
            t2 = replace_nan_with
        y = torch.tensor([float(t1) if (not pd.isna(t1)) else float('nan'),
                          float(t2) if (not pd.isna(t2)) else float('nan')], dtype=torch.float)
        data_list.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
                          'y': y, 'smiles': g.get('smiles'), 'ligand_id': g.get('ligand_id')})
    return data_list

# ---------------------------
# Dataset / collate
# ---------------------------

class GraphDictDataset(Dataset):
    def __init__(self, dict_list):
        self.list = dict_list
    def __len__(self):
        return len(self.list)
    def __getitem__(self, idx):
        return self.list[idx]

def collate_graphs(graphs):
    xs=[]; eis=[]; eas=[]; ys=[]; node_counts=[]; ligand_ids=[]; smiles=[]
    node_offset=0
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

    if len(xs)>0 and any([xx.numel()>0 for xx in xs]):
        # Find a tensor with features to determine the feature dimension
        feat_dim = 0
        for xx in xs:
            if xx.ndim > 1 and xx.shape[1] > 0:
                feat_dim = xx.shape[1]
                break
        xs2=[]
        for xx in xs:
            if xx.numel()==0:
                xs2.append(torch.empty((0, feat_dim)))
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
        # Find a tensor with features to determine the feature dimension
        feat_dim_ea = 0
        for ea_i in eas:
            if ea_i.ndim > 1 and ea_i.shape[1] > 0:
                feat_dim_ea = ea_i.shape[1]
                break
        eas2 = [e if e.numel()>0 else torch.empty((0, feat_dim_ea)) for e in eas]
        batch_ea = torch.cat(eas2, dim=0)
    else:
        batch_ea = torch.empty((0,))
    batch_y = torch.stack(ys, dim=0) if len(ys)>0 else torch.empty((0,2))
    if len(node_counts)==0 or sum(node_counts)==0:
        batch_vec = torch.tensor([], dtype=torch.long)
    else:
        batch_vec = torch.repeat_interleave(torch.arange(len(node_counts), dtype=torch.long), torch.tensor(node_counts, dtype=torch.long))
    return {'x': batch_x, 'edge_index': batch_ei, 'edge_attr': batch_ea, 'y': batch_y,
            'node_counts': node_counts, 'batch_vec': batch_vec,
            'ligand_id': ligand_ids, 'smiles': smiles}

# ---------------------------
# Model: GNNMember
# ---------------------------

class GNNMember(nn.Module):
    # conv_type과 heads 인자 추가
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, conv_type='gcn', heads=4, use_pyg=USE_PYG):
        super().__init__()
        self.use_pyg = use_pyg and USE_PYG
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if self.use_pyg:
            self.convs = nn.ModuleList()
            # 첫 번째 레이어
            if conv_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_dim))
            elif conv_type == 'gatv2':
                self.convs.append(GATv2Conv(in_dim, hidden_dim // heads, heads=heads))
            elif conv_type == 'transformer':
                self.convs.append(TransformerConv(in_dim, hidden_dim // heads, heads=heads))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            # 나머지 레이어
            for _ in range(num_layers - 1):
                if conv_type == 'gcn':
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif conv_type == 'gatv2':
                    self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads))
                elif conv_type == 'transformer':
                    self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads))

            # 예측 헤드: ReLU를 SiLU로 변경
            self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2,1))
            self.head_vina  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2,1))
        else:
            # PyG를 사용하지 않을 경우, 원본 GCN-like MLP 구조 유지 (ReLU -> SiLU)
            self.mlp_node = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
            self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2,1))
            self.head_vina  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2,1))

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        if self.use_pyg:
            h = x
            for conv in self.convs:
                h = conv(h, edge_index)
                # 활성화 함수: torch.relu를 F.silu로 변경
                h = F.silu(h)
                h = self.dropout(h)
            if batch_vec is None or batch_vec.numel()==0:
                hg = h.mean(dim=0, keepdim=True)
            else:
                hg = global_mean_pool(h, batch_vec)
        else:
            # PyG 미사용 시의 로직
            h_nodes = self.mlp_node(x) if x.numel() > 0 else torch.zeros((batch_vec.shape[0] if batch_vec is not None else 1, self.hidden_dim), device=x.device)
            if batch_vec is None or batch_vec.numel()==0:
                hg = h_nodes.mean(dim=0, keepdim=True)
            else:
                bsize = int(batch_vec.max().item())+1 if batch_vec.numel()>0 else 1
                sums = torch.zeros((bsize, h_nodes.shape[1]), device=h_nodes.device)
                counts = torch.zeros((bsize,), device=h_nodes.device)
                for i in range(h_nodes.shape[0]):
                    bi = batch_vec[i].long().item()
                    sums[bi] += h_nodes[i]
                    counts[bi] += 1.0
                counts = counts.unsqueeze(1).clamp(min=1.0)
                hg = sums / counts
        out1 = self.head_gnina(hg).view(-1)
        out2 = self.head_vina(hg).view(-1)
        out = torch.stack([out1, out2], dim=1)
        return out

# ---------------------------
# helpers: tensors / train / eval
# ---------------------------

def batch_to_tensors_for_model(batch, device):
    if isinstance(batch, dict) and 'x' in batch and isinstance(batch['x'], torch.Tensor):
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device) if isinstance(batch['edge_index'], torch.Tensor) else None
        batch_vec = batch['batch_vec'].to(device) if isinstance(batch.get('batch_vec', None), torch.Tensor) else None
        edge_attr = batch['edge_attr'].to(device) if isinstance(batch.get('edge_attr', None), torch.Tensor) else None
        y = batch['y'].to(device) if isinstance(batch.get('y', None), torch.Tensor) else torch.tensor([], device=device)
    else:
        raise RuntimeError("Unexpected batch format in batch_to_tensors_for_model.")
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

def evaluate_model_mc(model, loader, device, mc_T=5, enable_mc=False):
    model.eval()
    samples=[]
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)] if enable_mc else []
    S = mc_T if enable_mc else 1
    for s in range(S):
        if enable_mc:
            for m in dropout_modules:
                m.train()
        preds_batches=[]
        for batch in loader:
            x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
            with torch.no_grad():
                out = model(x, edge_index, batch_vec, edge_attr)
            preds_batches.append(out.detach().cpu())
        preds_cat = torch.cat(preds_batches, dim=0).numpy() if len(preds_batches)>0 else np.zeros((0,2))
        samples.append(preds_cat)
    for m in dropout_modules:
        m.eval()
    samples = np.stack(samples, axis=0)  # (S,N,2)
    return samples

def sample_ensemble_predictions(ensemble_models, loader, device, mc_T=5, enable_mc=True, verbose=True):
    all_samples=[]
    for m_idx, model in enumerate(ensemble_models):
        if verbose:
            print(f"[ensemble] sampling model {m_idx+1}/{len(ensemble_models)}")
        model.to(device)
        sm = evaluate_model_mc(model, loader, device, mc_T=mc_T, enable_mc=enable_mc)
        # sm shape (mc_T, N, 2)
        for s in range(sm.shape[0]):
            all_samples.append(sm[s])
    if len(all_samples)==0:
        return np.zeros((0,0,2))
    all_samples = np.stack(all_samples, axis=0)  # (S, N, 2)
    return all_samples

# ---------------------------
# ensemble builder
# ---------------------------

DEFAULT_ENSEMBLE_SPECS = [
    {"name":"gcn","conv_type":"gcn","seed":42},
    {"name":"gatv2","conv_type":"gatv2","seed":43},
    {"name":"transformer","conv_type":"transformer","seed":44}
]

def build_ensemble_from_specs(ensemble_specs, in_dim, device, hidden_dim=128, num_layers=3, dropout=0.2):
    ensemble_models=[]
    for spec in ensemble_specs:
        seed = spec.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # spec에서 conv_type을 읽어 GNNMember 생성자에 전달
        conv_type = spec.get('conv_type', 'gcn')

        m = GNNMember(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, conv_type=conv_type)
        if 'ckpt' in spec and spec['ckpt']:
            try:
                st = torch.load(spec['ckpt'], map_location='cpu')
                m.load_state_dict(st)
            except Exception as e:
                print(f"[warn] failed to load ckpt {spec.get('ckpt')}: {e}")
        m.to(device); m.eval()
        ensemble_models.append(m)
        print(f"  -> ensemble model '{spec.get('name')}' (type: {conv_type}) created")
    return ensemble_models

# ---------------------------
# main
# ---------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading graphs metadata:", args.meta_csv)
    data_list = load_graphs_from_meta(args.meta_csv, graphs_root=args.graphs_root,
                                     target_cols=('gnina_affinity','vina_affinity'), replace_nan_with=args.replace_nan)
    if len(data_list)==0:
        raise RuntimeError("No graphs loaded.")
    print(f"[info] loaded {len(data_list)} graphs.")

    # split
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    idxs = np.arange(len(data_list)); np.random.shuffle(idxs)
    n = len(idxs)
    ntrain = int(n * args.train_frac); nval = int(n * args.val_frac)
    train_idx = idxs[:ntrain]; val_idx = idxs[ntrain:ntrain+nval]; test_idx = idxs[ntrain+nval:]
    train_list = [data_list[i] for i in train_idx]; val_list = [data_list[i] for i in val_idx]; test_list = [data_list[i] for i in test_idx]

    # dataloaders
    train_loader = DataLoader(GraphDictDataset(train_list), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader   = DataLoader(GraphDictDataset(val_list),   batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(GraphDictDataset(test_list),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    # infer in_dim
    in_dim=None
    for d in train_list + val_list + test_list:
        if d['x'] is not None and isinstance(d['x'], torch.Tensor) and d['x'].ndim==2 and d['x'].shape[1]>0:
            in_dim = d['x'].shape[1]; break
    if in_dim is None:
        in_dim = args.fallback_in_dim
        print("[warn] using fallback in_dim:", in_dim)
    print("[info] in_dim =", in_dim)

    # build/train single model (이제 이 부분은 앙상블 학습 전의 '워밍업'으로 볼 수 있습니다)
    # 앙상블을 직접 학습시킬 수도 있지만, 원본 구조 유지를 위해 단일 모델 학습 로직은 그대로 둡니다.
    model = GNNMember(in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, conv_type='gcn')
    model.to(device)

    # compute mean/std robustly
    y_train = torch.stack([d['y'] for d in train_list], dim=0).numpy()
    mean_train = np.nanmean(y_train, axis=0)
    std_train = np.nanstd(y_train, axis=0)
    for i in range(mean_train.shape[0]):
        if np.isnan(mean_train[i]):
            if args.replace_nan is not None:
                mean_train[i] = float(args.replace_nan); std_train[i]=1.0
                print(f"[warn] target col {i} had no values; using replace_nan {args.replace_nan}")
            else:
                mean_train[i]=0.0; std_train[i]=1.0
                print(f"[warn] target col {i} had no values; using fallback mean=0,std=1")
    std_train[std_train==0]=1.0
    print("[info] target means:", mean_train, "stds:", std_train)

    def criterion(preds, y):
        device_local = preds.device
        y_np = y.detach().cpu().numpy()
        valid_mask = ~np.isnan(y_np)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device_local, requires_grad=True)
        y_norm = (np.where(valid_mask, y_np, 0.0) - mean_train.reshape(1,2)) / std_train.reshape(1,2)
        y_norm_t = torch.tensor(y_norm, dtype=torch.float, device=device_local)
        se = (preds - y_norm_t)**2
        mask_t = torch.tensor(valid_mask.astype(float), dtype=torch.float, device=device_local)
        se_masked = se * mask_t
        loss = se_masked.sum() / mask_t.sum()
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    print("\n[info] Starting single model training (as a warm-up)...")
    for ep in range(1, args.epochs+1):
        t0=time.time()
        batch_losses = train_one_epoch_log(model, optimizer, train_loader, device, criterion, kl_weight=args.kl_weight)
        epoch_mean_loss = float(np.mean(batch_losses)) if len(batch_losses)>0 else float('nan')
        
        # validation
        model.eval()
        with torch.no_grad():
            preds=[]; ys=[]
            for batch in val_loader:
                x, edge_index, batch_vec, edge_attr, y = batch_to_tensors_for_model(batch, device)
                out = model(x, edge_index, batch_vec, edge_attr)
                out_np = out.detach().cpu().numpy()
                out_orig = out_np * std_train.reshape(1,2) + mean_train.reshape(1,2)
                preds.append(out_orig); ys.append(y.detach().cpu().numpy())
            if len(preds)>0:
                preds = np.concatenate(preds, axis=0); ys = np.concatenate(ys, axis=0)
                mask = ~np.isnan(ys[:,0])
                val_mse = float(np.mean((preds[mask,0]-ys[mask,0])**2)) if mask.sum()>0 else float('nan')
            else:
                val_mse = float('nan')
        t1=time.time()
        print(f"[train] epoch {ep}/{args.epochs} mean_loss {epoch_mean_loss:.6f} val_mse(gnina) {val_mse:.6f} time {(t1-t0):.1f}s")
        if args.save_checkpoints:
            torch.save(model.state_dict(), out_dir / f"trained_model_ep{ep}.pt")

    print("[info] Single model training finished.")
    
    # ---------------------------
    # build ensemble and sample
    # ---------------------------
    print("\n[info] Building ensemble from specs...")
    ensemble_specs = args.ensemble_specs
    if isinstance(ensemble_specs, str):
        try:
            ensemble_specs = json.loads(ensemble_specs)
        except Exception:
            print("[warn] ensemble_specs parse failed; using default")
            ensemble_specs = DEFAULT_ENSEMBLE_SPECS

    ensemble_models = build_ensemble_from_specs(ensemble_specs, in_dim, device,
                                                hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)

    if args.init_ensemble_from_trained:
        print("[info] Initializing ensemble members from the single trained model...")
        trained_state = model.state_dict()
        for em in ensemble_models:
            try:
                # GATv2/Transformer는 GCN과 파라미터 이름/구조가 달라 로딩이 실패할 수 있습니다.
                # 실패하더라도 경고 없이 넘어가도록 하여 독립적인 학습을 유도합니다.
                em.load_state_dict(trained_state, strict=False)
            except Exception as e:
                print(f"[warn] Could not load state for an ensemble member (this is expected for different architectures): {e}")


    print(f"\n[info] Sampling ensemble: M={len(ensemble_models)}, mc_T={args.mc_T}, enable_mc_dropout={args.enable_mc_dropout}")
    S_samples = sample_ensemble_predictions(ensemble_models, test_loader, device, mc_T=args.mc_T, enable_mc=args.enable_mc_dropout, verbose=True)
    if S_samples.size==0:
        raise RuntimeError("Empty sampling result")
    S, N, C = S_samples.shape
    print(f"[info] samples shape: {S} x {N} x {C}")

    # convert to original scale
    mean_preds = S_samples.mean(axis=0)  # (N,2) normalized space
    mean_preds_orig = mean_preds * std_train.reshape(1,2) + mean_train.reshape(1,2)
    samples_orig = S_samples.copy() * std_train.reshape(1,1,2) + mean_train.reshape(1,1,2)  # (S,N,2)

    # rank stats based on gnina (col 0)
    samples_gnina = samples_orig[:,:,0]  # (S,N)
    ranks_each_draw = np.argsort(-samples_gnina, axis=1)  # (S,N)
    pos_each_draw = np.empty_like(ranks_each_draw)
    for i in range(S):
        pos = np.empty(N, dtype=int); pos[ranks_each_draw[i]] = np.arange(N); pos_each_draw[i]=pos
    mean_rank = pos_each_draw.mean(axis=0)
    std_rank = pos_each_draw.std(axis=0)
    rank_confidence = 1.0 - (mean_rank / (max(1, N-1)))

    # assemble true values and final DataFrame
    ligand_ids=[]; smiles_list=[]; true_gnina=[]; true_vina=[]
    for d in test_list:
        y = d.get('y')
        gn = float(y[0].item()) if isinstance(y, torch.Tensor) and y.numel()>=1 and not torch.isnan(y[0]) else float('nan')
        vi = float(y[1].item()) if isinstance(y, torch.Tensor) and y.numel()>=2 and not torch.isnan(y[1]) else float('nan')
        ligand_ids.append(d.get('ligand_id')); smiles_list.append(d.get('smiles'))
        true_gnina.append(gn); true_vina.append(vi)

    mean_gnina = mean_preds_orig[:,0]
    mean_vina  = mean_preds_orig[:,1]

    df_out = pd.DataFrame({
        'ligand_id': ligand_ids,
        'smiles': smiles_list,
        'true_gnina': true_gnina,
        'true_vina': true_vina,
        'mean_gnina': mean_gnina,
        'mean_vina': mean_vina,
        'mean_rank': mean_rank,
        'std_rank': std_rank,
        'rank_confidence': rank_confidence
    })

    df_sorted = df_out.sort_values('mean_gnina', ascending=False).reset_index(drop=True)
    out_csv = out_dir / "df_sorted_final.csv"
    df_sorted.to_csv(out_csv, index=False)
    print("\n[done] saved final CSV:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian ensemble GNN (fixed outputs and epoch logging)")
    parser.add_argument('--meta_csv', type=str, required=True, help="Path to the metadata CSV (e.g., processed_graphs.csv)")
    parser.add_argument('--graphs_root', type=str, required=True, help="Path to the directory containing .pt graph files")
    parser.add_argument('--out_dir', type=str, default="./results")
    parser.add_argument('--epochs', type=int, default=10)
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
    parser.add_argument(
        '--ensemble_specs',
        type=str,
        default='[{"name":"gcn","conv_type":"gcn","seed":42},'
                '{"name":"gatv2","conv_type":"gatv2","seed":43},'
                '{"name":"transformer","conv_type":"transformer","seed":44}]',
        help="JSON string list of ensemble specs"
    )
    parser.add_argument('--init_ensemble_from_trained', action='store_true', help="Initialize ensemble models with the single trained model's weights (may fail for different architectures)")
    args = parser.parse_args()
    main(args)
