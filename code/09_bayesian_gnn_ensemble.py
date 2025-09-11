#!/usr/bin/env python3
"""
09_bayesian_gnn_final.py

사용 예시:
python 09_bayesian_gnn_final.py --meta_csv 'path/to/your/metacsv.csv' --graphs_root 'path/to/your/graphs' --out_dir './final_results' --epochs 20
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

# torch_geometric 라이브러리가 있으면 사용
USE_PYG = False
try:
    from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
    USE_PYG = True
    print("[info] torch_geometric's GNN layers are available.")
except Exception:
    USE_PYG = False
    print("[warn] torch_geometric not found. Using a simple MLP-based model as a fallback.")

# -------------------------
# 유틸리티: 메타데이터 및 그래프 로드
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
        # 타겟 값 처리
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
# Dataset & Collate 함수
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
        if ei is None or (isinstance(ei, torch.Tensor) and ei.numel()==0):
            shifted_ei = torch.empty((2,0), dtype=torch.long)
        else:
            shifted_ei = (ei.clone() if isinstance(ei, torch.Tensor) else torch.tensor(ei, dtype=torch.long).t().contiguous()) + node_offset
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

    # 배치 텐서 생성
    batch_x = torch.cat(xs, dim=0) if xs and any(t.numel() > 0 for t in xs) else torch.empty(0)
    batch_ei = torch.cat(eis, dim=1) if eis and any(t.numel() > 0 for t in eis) else torch.empty(2, 0, dtype=torch.long)
    batch_ea = torch.cat([e for e in eas if e is not None and e.numel() > 0], dim=0) if eas and any(e is not None and e.numel() > 0 for e in eas) else torch.empty(0)
    batch_y = torch.stack(ys, dim=0) if ys else torch.empty(0, 2)
    batch_vec = torch.repeat_interleave(torch.arange(len(node_counts)), torch.tensor(node_counts, dtype=torch.long))
    
    return {'x': batch_x, 'edge_index': batch_ei, 'edge_attr': batch_ea, 'y': batch_y, 'node_counts': node_counts, 'batch_vec': batch_vec, 'ligand_id': ligand_ids, 'smiles': smiles}

# -------------------------
# 모델 정의 (GNNMember)
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
            ConvLayer = GCNConv # 기본값
            if 'gatv2' in self.conv_type: ConvLayer = GATv2Conv
            elif 'transformer' in self.conv_type: ConvLayer = TransformerConv
            
            self.convs.append(ConvLayer(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(ConvLayer(hidden_dim, hidden_dim))
        else:
            # PyG가 없을 경우, 간단한 MLP로 대체
            self.mlp_node = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        
        self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.head_vina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float, device=next(self.parameters()).device)
        
        if self.use_pyg:
            h = x
            for conv in self.convs:
                if 'transformer' in self.conv_type and edge_attr is not None and edge_attr.numel() > 0:
                    h = conv(h, edge_index, edge_attr)
                else:
                    h = conv(h, edge_index)
                h = torch.relu(h)
                h = self.dropout(h)
            hg = global_mean_pool(h, batch_vec) if batch_vec is not None and batch_vec.numel() > 0 else h.mean(dim=0, keepdim=True)
        else:
            # PyG가 없을 때: global_mean_pool 대신 수동으로 평균 계산 (오류 수정)
            h_nodes = self.mlp_node(x) if x.numel() > 0 else torch.zeros((0, self.hidden_dim), device=x.device)
            if batch_vec is None or batch_vec.numel() == 0:
                hg = h_nodes.mean(dim=0, keepdim=True)
            else:
                # 각 그래프의 노드 피처를 합산하여 평균을 계산
                num_graphs = batch_vec.max().item() + 1
                hg = torch.zeros(num_graphs, h_nodes.size(1), device=h_nodes.device)
                hg.index_add_(0, batch_vec, h_nodes) # batch_vec을 인덱스로 사용해 합산
                
                node_counts = torch.bincount(batch_vec, minlength=num_graphs).unsqueeze(1).clamp(min=1)
                hg = hg / node_counts
            
        out1 = self.head_gnina(hg).view(-d1)
        out2 = self.head_vina(hg).view(-1)
        return torch.stack([out1, out2], dim=1)

# -------------------------
# 헬퍼 함수
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

def build_ensemble_from_specs(specs, in_dim, device, hidden_dim, num_layers, dropout):
    models = []
    for spec in specs:
        seed = spec.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        conv_type = spec.get('conv_type', 'gcn')
        m = GNNMember(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, conv_type=conv_type)
        m.to(device)
        models.append(m)
    return models

# -------------------------
# 샘플링 함수
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
        if verbose: print(f"  Sampling from model {m_idx+1}/{len(ensemble_models)} ({conv_type.upper()}) (MC-T={mc_T if enable_mc else 1})")
        
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

    enable_mc_dropout(m, False) # 마지막 모델 드롭아웃 비활성화
    if not all_preds_draws: return np.zeros((0, 0, 0))
    return np.stack(all_preds_draws, axis=0) # (S, N, 2)

# -------------------------
# 아핀 변환 보정 함수
# -------------------------
def fit_affine_map(x, y, use_robust=False):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2: return 1.0, 0.0
    
    xs, ys = x[mask].reshape(-1, 1), y[mask]
    if use_robust:
        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(xs, ys)
            a, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            return float(a), float(b)
        except ImportError:
            print("[warn] scikit-learn not found for RANSAC. Falling back to OLS.")
    
    a, b = np.polyfit(xs.ravel(), ys, 1)
    return float(a), float(b)

# -------------------------
# 메인 함수
# -------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Using device: {device}")

    # 데이터 로드
    data_list = load_graphs_from_meta(args.meta_csv, graphs_root=args.graphs_root, replace_nan_with=args.replace_nan)
    if not data_list: raise RuntimeError("No data loaded.")
    print(f"[info] {len(data_list)} graphs loaded.")

    # 데이터 분할
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    idxs = np.arange(len(data_list)); np.random.shuffle(idxs)
    n = len(idxs)
    ntrain = int(n * args.train_frac); nval = int(n * args.val_frac)
    train_idx, val_idx, test_idx = idxs[:ntrain], idxs[ntrain:ntrain+nval], idxs[ntrain+nval:]
    train_list, val_list, test_list = [data_list[i] for i in train_idx], [data_list[i] for i in val_idx], [data_list[i] for i in test_idx]
    print(f"[info] Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    # 데이터로더 생성
    train_loader = DataLoader(GraphDictDataset(train_list), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    test_loader  = DataLoader(GraphDictDataset(test_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    # 입력 차원 결정
    in_dim = next((d['x'].shape[1] for d in data_list if d['x'] is not None and d['x'].ndim == 2 and d['x'].shape[1] > 0), args.fallback_in_dim)
    print(f"[info] Determined input feature dimension (in_dim): {in_dim}")

    # 학습 데이터의 타겟 평균/표준편차 계산 (정규화용)
    y_train = np.array([d['y'].numpy() for d in train_list])
    mean_train = np.nanmean(y_train, axis=0, keepdims=True)
    std_train = np.nanstd(y_train, axis=0, keepdims=True)
    std_train = np.where(std_train == 0, 1.0, std_train)
    print(f"[info] Train target mean: {mean_train.ravel()}, std: {std_train.ravel()}")

    # 손실 함수 정의 (정규화된 값 기준)
    def criterion(preds, y):
        y_norm = (y - torch.tensor(mean_train, dtype=torch.float, device=preds.device)) / torch.tensor(std_train, dtype=torch.float, device=preds.device)
        mask = ~torch.isnan(y_norm)
        if not mask.any(): return torch.tensor(0.0, device=preds.device, requires_grad=True)
        loss = nn.functional.mse_loss(preds[mask], y_norm[mask])
        return loss

    # 앙상블 모델 빌드
    try:
        ensemble_specs = json.loads(args.ensemble_specs)
    except json.JSONDecodeError:
        print("[warn] Failed to parse ensemble_specs. Using default.")
        ensemble_specs = [
            {'name':'gcn','conv_type':'gcn','seed':42},
            {'name':'gatv2','conv_type':'gatv2','seed':43},
            {'name':'transformer','conv_type':'transformer','seed':44}
        ]
    ensemble_models = build_ensemble_from_specs(ensemble_specs, in_dim, device, args.hidden_dim, args.num_layers, args.dropout)
    print(f"[info] Built ensemble with {len(ensemble_models)} members.")

    # **앙상블의 각 모델을 개별적으로 학습**
    for i, model in enumerate(ensemble_models):
        spec = ensemble_specs[i]
        print(f"\n--- Training ensemble member {i+1}/{len(ensemble_models)}: {spec.get('name', 'GNN')} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for ep in range(1, args.epochs + 1):
            batch_losses = train_one_epoch_log(model, optimizer, train_loader, device, criterion)
            print(f"[train] Ep {ep}/{args.epochs}, Mean Loss: {np.mean(batch_losses):.6f}")
        if args.save_checkpoints:
            torch.save(model.state_dict(), out_dir / f"ensemble_member_{i}_{spec.get('name', 'model')}.pt")
    print("\n--- All ensemble members trained. ---\n")

    # 앙상블 샘플링
    print("[info] Sampling predictions from the trained ensemble...")
    S_samples = sample_ensemble_predictions(ensemble_models, test_loader, device, args.mc_T, args.enable_mc_dropout, verbose=True)
    if S_samples.size == 0: raise RuntimeError("Sampling returned no predictions.")
    S, N, C = S_samples.shape
    print(f"[info] Sampling complete. Got tensor of shape (S, N, C): ({S}, {N}, {C})")

    # 예측값 후처리
    samples_orig = S_samples * std_train + mean_train
    mean_preds_orig = samples_orig.mean(axis=0)
    
    true_gnina = np.array([d['y'][0].item() if d['y'].numel() >= 1 else np.nan for d in test_list])
    true_vina = np.array([d['y'][1].item() if d['y'].numel() >= 2 else np.nan for d in test_list])

    pred_g, pred_v = mean_preds_orig[:, 0], mean_preds_orig[:, 1]

    # Min-Max 리스케일링
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
        pred_g_rescaled, pred_v_rescaled = pred_g, pred_v

    # 아핀 변환 보정
    if args.calibrate_affine:
        a_g, b_g = fit_affine_map(pred_g_rescaled, true_gnina, args.use_robust_calib)
        pred_g_cal = a_g * pred_g_rescaled + b_g
        a_v, b_v = fit_affine_map(pred_v_rescaled, true_vina, args.use_robust_calib)
        print("[info] Applied affine calibration.")
        try:
            mask_g = ~np.isnan(true_gnina) & ~np.isnan(pred_g_cal)
            if mask_g.sum() > 1:
                pearson_before = pearsonr(pred_g_rescaled[mask_g], true_gnina[mask_g])[0]
                pearson_after = pearsonr(pred_g_cal[mask_g], true_gnina[mask_g])[0]
                print(f"[calib] GNINA: a={a_g:.4f}, b={b_g:.4f}. Pearson (before/after): {pearson_before:.4f} -> {pearson_after:.4f}")
        except Exception: pass
    else:
        pred_g_cal, pred_v_cal = pred_g_rescaled, pred_v_rescaled
        a_g, b_g, a_v, b_v = 1.0, 0.0, 1.0, 0.0

    # 불확실성 및 순위 통계 계산
    std_per_item_g = np.nanstd(samples_orig[:, :, 0], axis=0)
    std_per_item_v = np.nanstd(samples_orig[:, :, 1], axis=0)

    score = pred_g_cal - args.uncertainty_lambda * std_per_item_g if args.uncertainty_lambda > 0 else pred_g_cal
    
    ranks_s = np.argsort(-samples_orig[:, :, 0], axis=1).argsort(axis=1)
    mean_rank, std_rank = ranks_s.mean(axis=0), ranks_s.std(axis=0)
    rank_confidence = 1.0 - (mean_rank / (max(1, N - 1)))
    
    # 최종 데이터프레임 생성 및 저장
    df_out = pd.DataFrame({
        'ligand_id': [d.get('ligand_id') for d in test_list],
        'smiles': [d.get('smiles') for d in test_list],
        'true_gnina': true_gnina, 'true_vina': true_vina,
        'mean_gnina': pred_g, 'std_gnina': std_per_item_g,
        'mean_gnina_rescaled': pred_g_rescaled, 'mean_gnina_calibrated': pred_g_cal,
        'final_score': score,
        'mean_vina': pred_v, 'std_vina': std_per_item_v,
        'mean_rank': mean_rank, 'std_rank': std_rank, 'rank_confidence': rank_confidence,
        'calib_a_g': a_g, 'calib_b_g': b_g
    })
    
    df_sorted = df_out.sort_values('final_score', ascending=False).reset_index(drop=True)
    out_csv = out_dir / "df_sorted_final.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n[done] Saved final sorted results to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Bayesian GNN Ensemble Training and Calibration Pipeline")
    
    # 필수 인자 -> 기본값으로 변경
    parser.add_argument('--meta_csv', type=str, 
                        default='/home/ssm-user/project/scores/processed_graphs.csv', 
                        help="Path to the metadata CSV file.")
    parser.add_argument('--graphs_root', type=str, 
                        default='/home/ssm-user/project/graphs', 
                        help="Directory containing the graph .pt files.")
    parser.add_argument('--out_dir', type=str, 
                        default='/home/ssm-user/project/scores/results', 
                        help="Directory to save results.")
    
    # 학습 관련 인자 -> 요청하신 기본값으로 변경
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs for each ensemble member.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # 모델 구조 인자
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fallback_in_dim', type=int, default=52, help="Fallback input dimension if it cannot be inferred.")
    
    # 데이터 처리 인자
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1, help="Note: Validation set is defined but not used for early stopping in this script.")
    parser.add_argument('--replace_nan', type=float, default=None, help="Value to replace NaNs in target columns.")
    
    # 앙상블 및 샘플링 인자
    parser.add_argument('--ensemble_specs', type=str, default='[{"name":"gcn","conv_type":"gcn","seed":42},{"name":"gatv2","conv_type":"gatv2","seed":43},{"name":"transformer","conv_type":"transformer","seed":44}]', help="JSON string defining the ensemble members.")
    parser.add_argument('--mc_T', type=int, default=5, help="Number of Monte Carlo samples per model.")
    parser.add_argument('--enable_mc_dropout', action='store_true', help="Enable MC dropout for sampling.")
    
    # 후처리 및 보정 인자
    parser.add_argument('--rescale', type=str, choices=['none','minmax'], default='minmax', help="Rescaling method for predictions.")
    parser.add_argument('--calibrate_affine', action='store_true', default=True, help="Apply affine calibration (pred -> true).")
    parser.add_argument('--use_robust_calib', action='store_true', help="Use RANSAC for robust calibration if scikit-learn is available.")
    parser.add_argument('--uncertainty_lambda', type=float, default=0.5, help="Lambda for uncertainty penalty in final score (score = pred - lambda*std). Set to 0 to disable.")
    
    # 기타
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_checkpoints', action='store_true', help="Save model checkpoints after training each ensemble member.")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA, even if available.")
    
    args = parser.parse_args()
    main(args)
