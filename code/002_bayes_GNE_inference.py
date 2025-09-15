#!/usr/bin/env python3
"""
Command Example:
    python 10_predict_with_ensemble.py

If you find:
  [warn] torch_geometric not found. Using a simple MLP-based model as a fallback.
Then :
  Conda install -c conda-forge torch_geometric -y

"""

# -------------------------
# Library
# -------------------------

import os
import json
import argparse
from pathlib import Path
import random
import time 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import necessary components from the original script
# Ensure these are compatible if you've made custom changes.
try:
    from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, GINEConv, AttentionalAggregation
    USE_PYG = True
    print("[info] torch_geometric's GNN layers are available.")
except ImportError:
    USE_PYG = False
    print("[warn] torch_geometric not found. Using a simple MLP-based model as a fallback.")

# -------------------------
# Data Loading (from original script)
# -------------------------

def load_graphs_from_meta(meta_csv, graphs_root=None):
    """Loads graph data based on a metadata CSV file."""
    df = pd.read_csv(meta_csv)
    if graphs_root is not None:
        df['graph_path'] = df['graph_path'].apply(lambda p: os.path.join(graphs_root, os.path.basename(p)))
    
    data_list = []
    for _, row in df.iterrows():
        gpath = row['graph_path']
        if not os.path.exists(gpath):
            print(f"[warn] Missing graph file: {gpath}, skipping.")
            continue
        try:
            g = torch.load(gpath)
            # Ensure essential keys exist
            data_entry = {
                'x': g.get('node_attr'),
                'edge_index': g.get('edge_index'),
                'edge_attr': g.get('edge_attr', None),
                'smiles': g.get('smiles'),
                'ligand_id': g.get('ligand_id')
            }
            data_list.append(data_entry)
        except Exception as e:
            print(f"[error] Failed to load or process graph {gpath}: {e}")

    return data_list

# -------------------------
# Dataset & Collate (from original script)
# -------------------------

class GraphDictDataset(Dataset):
    def __init__(self, dict_list):
        self.list = dict_list
    def __len__(self):
        return len(self.list)
    def __getitem__(self, idx):
        return self.list[idx]

def collate_graphs(graphs):
    """Collates a list of graph dictionaries into a single batch dictionary."""
    xs, eis, eas, node_counts, ligand_ids, smiles = [], [], [], [], [], []
    node_offset = 0
    for g in graphs:
        x = g['x']
        if x is None: x = torch.empty((0, 0))
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float)
        xs.append(x)
        
        ei = g.get('edge_index')
        shifted_ei = (ei.clone() if isinstance(ei, torch.Tensor) and ei.numel() > 0 else torch.empty((2, 0), dtype=torch.long)) + node_offset
        eis.append(shifted_ei)

        ea = g.get('edge_attr')
        if ea is not None and not isinstance(ea, torch.Tensor): ea = torch.tensor(ea, dtype=torch.float)
        eas.append(ea)
        
        num_nodes = x.shape[0] if hasattr(x, 'shape') else 0
        node_counts.append(num_nodes)
        ligand_ids.append(g.get('ligand_id'))
        smiles.append(g.get('smiles'))
        node_offset += num_nodes

    batch_x = torch.cat(xs, dim=0) if xs and any(t.numel() > 0 for t in xs) else torch.empty(0)
    batch_ei = torch.cat(eis, dim=1) if eis and any(t.numel() > 0 for t in eis) else torch.empty(2, 0, dtype=torch.long)
    batch_ea = torch.cat([e for e in eas if e is not None and e.numel() > 0], dim=0) if eas and any(e is not None and e.numel() > 0 for e in eas) else torch.empty(0)
    batch_vec = torch.repeat_interleave(torch.arange(len(node_counts)), torch.tensor(node_counts, dtype=torch.long)) if node_counts else torch.empty(0, dtype=torch.long)
    
    return {'x': batch_x, 'edge_index': batch_ei, 'edge_attr': batch_ea, 'batch_vec': batch_vec, 'ligand_id': ligand_ids, 'smiles': smiles}

# -------------------------
# GNN Model (from original script)
# -------------------------

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
                if edge_dim is None: raise ValueError("edge_dim must be specified for GINEConv")
                self.convs.append(GINEConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.BatchNorm1d(hidden_dim)), edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.BatchNorm1d(hidden_dim)), edge_dim=edge_dim))

            elif 'transformer' in self.conv_type:
                if edge_dim is None: raise ValueError("edge_dim must be specified for TransformerConv")
                self.convs.append(TransformerConv(in_dim, hidden_dim, edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(TransformerConv(hidden_dim, hidden_dim, edge_dim=edge_dim))            
            
            else:
                ConvLayer = {'gcn': GCNConv, 'gatv2': GATv2Conv}.get(self.conv_type, GCNConv)
                self.convs.append(ConvLayer(in_dim, hidden_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(ConvLayer(hidden_dim, hidden_dim))
            
            self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        else:
            self.mlp_node = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        
        self.head_gnina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2, 1))
        self.head_vina = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2, 1))

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
        else: # Fallback
            h_nodes = self.mlp_node(x)
            num_graphs = batch_vec.max().item() + 1
            hg = torch.zeros(num_graphs, h_nodes.size(1), device=h_nodes.device)
            hg.index_add_(0, batch_vec, h_nodes)
            node_counts = torch.bincount(batch_vec, minlength=num_graphs).unsqueeze(1).clamp(min=1)
            hg = hg / node_counts
            
        out1 = self.head_gnina(hg).view(-1)
        out2 = self.head_vina(hg).view(-1)
        return torch.stack([out1, out2], dim=1)

# -------------------------
# Inference Helpers
# -------------------------

def enable_mc_dropout(model, enable=True):
    """Enable or disable dropout layers for Monte-Carlo sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(enable)

def predict_with_ensemble(ensemble_models, loader, device, mc_T=5, verbose=False):
    """Generates predictions using the ensemble and MC dropout."""
    all_preds_draws = []
    
    ligand_ids = []
    smiles_list = []
    
    # Store identifiers once
    for batch in loader:
        ligand_ids.extend(batch['ligand_id'])
        smiles_list.extend(batch['smiles'])

    for m_idx, m in enumerate(ensemble_models):
        m.eval() # Ensure model is in evaluation mode
        conv_type = m.conv_type if hasattr(m, 'conv_type') else 'unknown'
        if verbose: print(f"  > Sampling from model {m_idx+1}/{len(ensemble_models)} ({conv_type.upper()}) (MC-T={mc_T})")
        
        for draw in range(mc_T):
            enable_mc_dropout(m, enable=True) # Enable dropout for this MC sample
            preds_list = []
            with torch.no_grad():
                for batch in loader:
                    x = batch['x'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    batch_vec = batch['batch_vec'].to(device)
                    edge_attr = batch.get('edge_attr')
                    edge_attr = edge_attr.to(device) if edge_attr is not None and edge_attr.numel() > 0 else None
                    
                    out = m(x, edge_index, batch_vec, edge_attr)
                    preds_list.append(out.cpu().numpy())
            preds_cat = np.concatenate(preds_list, axis=0)
            all_preds_draws.append(preds_cat)

    enable_mc_dropout(m, False) # Reset dropout to eval mode
    if not all_preds_draws: return np.zeros((0, 0, 0)), [], []
    
    return np.stack(all_preds_draws, axis=0), ligand_ids, smiles_list

# -------------------------
# Main Inference Pipeline
# -------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Using device: {device}")
    print(f"[info] Output will be saved to: {out_file}")

    # 1. Load all data for prediction
    data_list = load_graphs_from_meta(args.meta_csv, graphs_root=args.graphs_root)
    if not data_list:
        raise RuntimeError("No data loaded. Check --meta_csv and --graphs_root paths.")
    print(f"[info] Loaded {len(data_list)} graphs for prediction.")

    loader = DataLoader(GraphDictDataset(data_list), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    # 2. Determine model dimensions from data
    in_dim = next((d['x'].shape[1] for d in data_list if d['x'] is not None and d['x'].ndim == 2), args.fallback_in_dim)
    edge_dim = next((d['edge_attr'].shape[1] for d in data_list if d.get('edge_attr') is not None and d['edge_attr'].ndim == 2), None)
    print(f"[info] Inferred input feature dim: {in_dim}, Edge feature dim: {edge_dim}")

    # 3. Build and load pre-trained models
    ensemble_specs = json.loads(args.ensemble_specs)
    ensemble_models = []
    print("[info] Loading pre-trained ensemble models...")
    for spec in ensemble_specs:
        model_name = spec.get('name')
        conv_type = spec.get('conv_type')
        # Find the model file. Assumes a naming convention like 'ensemble_member_*_{name}_best.pt'
        model_path = next(Path(args.models_dir).glob(f"*_{model_name}_best.pt"), None)
        
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Could not find a pre-trained model file for '{model_name}' in '{args.models_dir}'.")

        print(f"  > Loading '{model_name}' from {model_path}")
        model = GNNMember(in_dim, args.hidden_dim, args.num_layers, args.dropout, conv_type, edge_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        ensemble_models.append(model)
        
    if len(ensemble_models) != len(ensemble_specs):
        raise RuntimeError("Failed to load all specified ensemble models.")
    print(f"[info] Successfully loaded {len(ensemble_models)} models.")

    # 4. Perform prediction
    print("\n[info] Starting prediction with ensemble...")
    inference_start_time = time.time()
    S_samples, ligand_ids, smiles = predict_with_ensemble(ensemble_models, loader, device, args.mc_T, verbose=True)
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
  
    if S_samples.size == 0:
        raise RuntimeError("Prediction failed, no samples were generated.")
    
    S, N, C = S_samples.shape # S = samples, N = ligands, C = targets (2)
    print(f"[info] Prediction complete. Generated {S} samples for {N} ligands.")

    # 5. Process predictions to calculate final scores and confidence
    print("[info] Calculating final scores and confidence metrics...")
    
    # GNINA is target 0, VINA is target 1
    gnina_samples = S_samples[:, :, 0]
    vina_samples = S_samples[:, :, 1]

    mean_gnina = np.mean(gnina_samples, axis=0)
    std_gnina = np.std(gnina_samples, axis=0)
    
    mean_vina = np.mean(vina_samples, axis=0)
    std_vina = np.std(vina_samples, axis=0)
    
    # Calculate final score (lower is better, penalizing uncertainty)
    # This is based on the logic from the training script
    final_score = mean_gnina - args.uncertainty_lambda * std_gnina

    # Calculate rank confidence
    rank_calc_start_time = time.time()
    # Calculate rank confidence
    ranks_s = np.argsort(-gnina_samples, axis=1).argsort(axis=1)
    mean_rank = ranks_s.mean(axis=0)
    std_rank = ranks_s.std(axis=0)
    rank_confidence = 1.0 - (mean_rank / (max(1, N - 1)))
    rank_calc_end_time = time.time()
    rank_calc_duration = rank_calc_end_time - rank_calc_start_time

    # 6. Create and save the final DataFrame
    df_out = pd.DataFrame({
        'ligand_id': ligand_ids,
        'smiles': smiles,
        'predicted_gnina_mean': mean_gnina,
        'predicted_gnina_std': std_gnina,
        'predicted_vina_mean': mean_vina,
        'predicted_vina_std': std_vina,
        'final_score': final_score,
        'rank_confidence': rank_confidence,
        'mean_rank': mean_rank,
        'std_rank': std_rank,
    })

    df_sorted = df_out.sort_values('final_score', ascending=True).reset_index(drop=True)
    df_sorted.to_csv(out_file, index=False)
    
    print("\n-----------------[ Summary ]---------------------")
    print(f"Successfully predicted ligands: {N}")
    print(f"Total inference time: {inference_duration:.2f} seconds")
    print(f"Rank confidence calculation time: {rank_calc_duration:.4f} seconds")
    print("---------------------------------------------------")
    
    print(f"\n[SUCCESS] Saved final sorted predictions to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with a pre-trained Bayesian GNN Ensemble.")
    
    # --- Paths (now with default values) ---
    parser.add_argument('--meta_csv', type=str, 
                        default='/home/ssm-user/project/scores/enamine_processed_graphs.csv',
                        help="Path to the metadata CSV for all ligands to be predicted.")
    parser.add_argument('--graphs_root', type=str, 
                        default='/home/ssm-user/project/scores/enamin_graphs',
                        help="Directory containing the graph .pt files.")
    parser.add_argument('--models_dir', type=str, 
                        default='/home/ssm-user/project/scores/trained_models',
                        help="Directory containing the 4 pre-trained model .pt files.")
    parser.add_argument('--out_file', type=str, 
                        default='/home/ssm-user/project/scores/results/all_ligands_predictions.csv',
                        help="Path to save the final output CSV file.")
    
    # --- Model Config (should match the trained models) ---
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension size of the models.")
    parser.add_argument('--num_layers', type=int, default=5, help="Number of GNN layers in the models.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate used during MC sampling.")
    parser.add_argument('--fallback_in_dim', type=int, default=111, help="Fallback input dim if it cannot be inferred.")
    parser.add_argument('--ensemble_specs', type=str,
                        default='[{"name":"gine","conv_type":"gine"},{"name":"gatv2","conv_type":"gatv2"},{"name":"transformer","conv_type":"transformer"},{"name":"gcn","conv_type":"gcn"}]',
                        help="JSON string defining the ensemble members. Names must match model files.")

    # --- Inference Config ---
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for inference (can be larger than training).")
    parser.add_argument('--mc_T', type=int, default=5, help="Number of Monte-Carlo samples per model.")
    parser.add_argument('--uncertainty_lambda', type=float, default=0.5, help="Weight for uncertainty penalty in the final score.")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA, use CPU instead.")
    
    args = parser.parse_args()
    main(args)
