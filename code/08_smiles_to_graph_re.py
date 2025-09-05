#!/usr/bin/env python3
"""
build_graphs_and_load_for_gnn.py

Single-file pipeline that:
- scans a directory for batch*_score.txt files (as in your example),
- parses SMILES rows and converts each ligand to a graph (.pt saved)
- writes a metadata CSV with paths and affinities
- provides a helper function to load the saved graphs into PyG Data objects (or simple dicts) ready for GNN training

Usage:
    python build_graphs_and_load_for_gnn.py --sdf_dir /path/to/scores --use_pos False

Requirements: RDKit, pandas, torch. Optional: torch_geometric for Data objects / batching.
"""

import os
import glob
import argparse
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# ---- Minimal gnina-like converter (keeps node/edge attrs) ----

def _is_rotatable_bond(bond, mol):
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE: return False
    if bond.IsInRing(): return False
    a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
    if a1.GetDegree() == 1 or a2.GetDegree() == 1: return False
    return True


def convert_mol_to_graph_gnina_like(mol, use_pos=False):
    mol2 = Chem.RemoveHs(mol)
    n_bonds = len(mol2.GetBonds())
    n_atoms = len(mol2.GetAtoms())

    edge_index = []
    edge_attr = []
    edge_weight = []

    for edge_idx in range(n_bonds):
        bond = mol2.GetBondWithIdx(edge_idx)
        a = bond.GetBeginAtomIdx(); b = bond.GetEndAtomIdx()
        edge_index.append([a,b]); edge_index.append([b,a])

        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bond_one_hot = [1,0,0,0]; edge_weight.extend([1.0,1.0])
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bond_one_hot = [0,1,0,0]; edge_weight.extend([1.5,1.5])
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bond_one_hot = [0,0,1,0]; edge_weight.extend([2.0,2.0])
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bond_one_hot = [0,0,0,1]; edge_weight.extend([3.0,3.0])
        else:
            bond_one_hot = [0,0,0,0]; edge_weight.extend([1.0,1.0])

        stype = bond.GetStereo(); stereo_one_hot = [0]*6
        if stype == Chem.rdchem.BondStereo.STEREOANY: stereo_one_hot[0]=1
        elif stype == Chem.rdchem.BondStereo.STEREOCIS: stereo_one_hot[1]=1
        elif stype == Chem.rdchem.BondStereo.STEREOE: stereo_one_hot[2]=1
        elif stype == Chem.rdchem.BondStereo.STEREONONE: stereo_one_hot[3]=1
        elif stype == Chem.rdchem.BondStereo.STEREOTRANS: stereo_one_hot[4]=1
        elif stype == Chem.rdchem.BondStereo.STEREOZ: stereo_one_hot[5]=1

        ring_bond = 1 if bond.IsInRing() else 0
        conjugate = 1 if bond.GetIsConjugated() else 0

        attr = bond_one_hot + stereo_one_hot + [ring_bond, conjugate]
        edge_attr.append(attr); edge_attr.append(attr)

    # node features
    valid_atoms = {
      'H':0,'B':1,'C':2,'N':3,'O':4,'F':5,'P':6,'S':7,'Cl':8,
      'Br':9,'I':10,'Fe':11,'Zn':12,'Mg':13,'Ca':14,'Na':15,'K':16,'OTHER':17
    }
    metals = {'Fe','Zn','Mg','Ca','Na','K','Cu','Mn','Co','Ni'}
    halogens = {'F','Cl','Br','I'}

    try:
        AllChem.ComputeGasteigerCharges(mol2)
        has_gasteiger = True
    except Exception:
        has_gasteiger = False

    node_attr = []
    for atm_id in range(n_atoms):
        atm = mol2.GetAtomWithIdx(atm_id)
        sym = atm.GetSymbol()
        atm_one_hot = [0]*len(valid_atoms)
        idx = valid_atoms.get(sym, valid_atoms['OTHER']); atm_one_hot[idx] = 1

        hybrid = atm.GetHybridization(); hybrid_one_hot = [0]*7
        if hybrid == Chem.HybridizationType.SP3: hybrid_one_hot[0]=1
        elif hybrid == Chem.HybridizationType.SP2: hybrid_one_hot[1]=1
        elif hybrid == Chem.HybridizationType.SP: hybrid_one_hot[2]=1
        elif hybrid == Chem.HybridizationType.S: hybrid_one_hot[3]=1
        elif hybrid == Chem.HybridizationType.SP3D: hybrid_one_hot[4]=1
        elif hybrid == Chem.HybridizationType.SP3D2: hybrid_one_hot[5]=1
        else: hybrid_one_hot[6]=1

        arom = 1 if atm.GetIsAromatic() else 0
        ring_flag = 1 if atm.IsInRing() else 0
        degree = atm.GetTotalDegree(); degree_one_hot = [0]*6; degree_one_hot[5 if degree>=5 else degree] = 1
        num_h = atm.GetTotalNumHs(); hydrogen_one_hot=[0]*5; hydrogen_one_hot[4 if num_h>=4 else num_h] = 1
        chiral = atm.GetChiralTag()
        if chiral == Chem.rdchem.ChiralType.CHI_OTHER: chiral_one_hot=[1,0,0,0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_one_hot=[0,1,0,0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_one_hot=[0,0,1,0]
        else: chiral_one_hot=[0,0,0,1]

        sym_upper = sym
        is_hbd = 0
        if sym_upper in {'N','O','S'} and atm.GetTotalNumHs()>0: is_hbd = 1
        is_hba = 1 if sym_upper in {'N','O','F','S','P','Cl'} else 0
        is_halogen = 1 if sym_upper in halogens else 0
        is_metal = 1 if sym_upper in metals else 0

        neighbor_polar = 0
        for neigh in atm.GetNeighbors():
            if neigh.GetSymbol() in {'N','O','F','Cl','S','P'}:
                neighbor_polar = 1; break

        g_charge = 0.0
        if has_gasteiger:
            try: g_charge = float(atm.GetProp('_GasteigerCharge'))
            except Exception: g_charge = 0.0

        atomic_num = atm.GetAtomicNum(); atomic_mass = atm.GetMass()

        attr = []
        attr += atm_one_hot
        attr += hybrid_one_hot
        attr += degree_one_hot
        attr += hydrogen_one_hot
        attr += chiral_one_hot
        attr += [arom, ring_flag, atm.GetFormalCharge(), atm.GetNumRadicalElectrons()]
        attr += [is_hbd, is_hba, is_halogen, is_metal, neighbor_polar]
        attr += [g_charge, float(atomic_num), float(atomic_mass)]

        node_attr.append(attr)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if len(edge_attr)>0 else torch.empty((0,))
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if len(edge_index)>0 else torch.empty((2,0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float) if len(edge_weight)>0 else torch.empty((0,))

    if use_pos:
        val = AllChem.EmbedMolecule(mol2)
        if val != 0:
            print(f"Error while generating 3D: {Chem.MolToSmiles(mol)}")
            return None
        pos_list = []; conf = mol2.GetConformer(0)
        for atm_id in range(n_atoms):
            p = conf.GetAtomPosition(atm_id); pos_list.append([p.x,p.y,p.z])
        pos = torch.tensor(pos_list, dtype=torch.float)
    else:
        pos = None

    return edge_index, node_attr, edge_attr, pos, edge_weight

# ---- Main pipeline to build graphs from batch files ----

def build_graphs_from_batch_files(SDF_DIR, out_csv_path=None, graphs_dir=None, use_pos=False):
    if out_csv_path is None: out_csv_path = os.path.join(SDF_DIR, "processed_graphs.csv")
    if graphs_dir is None: graphs_dir = os.path.join(SDF_DIR, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    file_pattern = os.path.join(SDF_DIR, "batch*_score.txt")
    files = sorted(glob.glob(file_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

    rows_out = []
    total = 0; kept = 0
    for fpath in files:
        batch_name = os.path.basename(fpath); print(f"Processing {batch_name} ...")
        try: df = pd.read_csv(fpath, sep=None, engine='python')
        except Exception: df = pd.read_csv(fpath, sep='\t', engine='python')

        columns_lower = {c.lower(): c for c in df.columns}
        smiles_col = None
        for candidate in ['smiles','smile','SMILES']:
            if candidate in columns_lower: smiles_col = columns_lower[candidate]; break
        if smiles_col is None:
            for c in df.columns:
                if 'smiles' in c.lower(): smiles_col = c; break
        if smiles_col is None:
            raise ValueError(f"Could not find SMILES column in {fpath}. Columns: {df.columns.tolist()}")

        ba_col = None; cnn_col = None
        for c in df.columns:
            if 'binding' in c.lower() and 'kcal' in c.lower(): ba_col = c
            if 'cnn' in c.lower() and 'aff' in c.lower(): cnn_col = c

        for idx, row in df.iterrows():
            total += 1
            smiles = str(row[smiles_col]).strip()
            ligand_id = row.get('Ligand', row.get('ligand', f"{os.path.splitext(batch_name)[0]}_{idx}"))
            ba_val = row.get(ba_col, None) if ba_col is not None else None
            cnn_val = row.get(cnn_col, None) if cnn_col is not None else None

            mol = None
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        try: Chem.SanitizeMol(mol)
                        except Exception: mol = None
            except Exception:
                mol = None

            if mol is None:
                try:
                    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
                    if tmp is not None:
                        Chem.SanitizeMol(tmp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                        mol = tmp
                except Exception:
                    mol = None

            if mol is None: continue

            try:
                res = convert_mol_to_graph_gnina_like(mol, use_pos=use_pos)
                if res is None: continue
                edge_index, node_attr, edge_attr, pos, edge_weight = res
            except Exception as e:
                print(f"Conversion error for {ligand_id} (idx {idx}) : {e}"); continue

            safe_name = f"{os.path.splitext(batch_name)[0]}_{idx}_{str(ligand_id)}".replace(os.sep, "_").replace(" ", "_")
            graph_path = os.path.join(graphs_dir, safe_name + ".pt")
            graph_obj = {
                "edge_index": edge_index.cpu() if isinstance(edge_index, torch.Tensor) else edge_index,
                "node_attr": node_attr.cpu(),
                "edge_attr": edge_attr.cpu() if isinstance(edge_attr, torch.Tensor) else edge_attr,
                "pos": pos.cpu() if isinstance(pos, torch.Tensor) else pos,
                "edge_weight": edge_weight.cpu() if isinstance(edge_weight, torch.Tensor) else edge_weight,
                "smiles": smiles,
                "ligand_id": ligand_id
            }
            torch.save(graph_obj, graph_path)

            rows_out.append({
                "batch_file": batch_name,
                "batch_index": idx,
                "ligand_id": ligand_id,
                "smiles": smiles,
                "graph_path": graph_path,
                "vina_affinity": ba_val,
                "gnina_affinity": cnn_val
            })
            kept += 1

    print(f"Processed total rows: {total}, kept graphs: {kept}")
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv_path, index=False)
    print(f"Saved metadata CSV to: {out_csv_path}")
    return out_df

# ---- helper to load saved graphs into structures ready for GNN ----

def load_graphs_for_gnn(meta_csv, graphs_root=None, target_col='gnina_affinity', replace_nan_with=None, to_pyg=True):
    """
    Loads saved graph .pt objects (written by build_graphs_from_batch_files) and returns:
      - if to_pyg and torch_geometric is available: list of torch_geometric.data.Data objects
      - else: list of dicts with keys: x (node_attr tensor), edge_index (tensor), edge_attr (tensor or None), y (tensor)

    meta_csv: path to the metadata CSV produced by build_graphs_from_batch_files
    graphs_root: optional root to prepend to graph_path entries
    target_col: name of column with the target (cnn affinity). If values missing, replace_nan_with can be used (float)
    """
    df = pd.read_csv(meta_csv)
    if graphs_root is not None:
        df['graph_path'] = df['graph_path'].apply(lambda p: os.path.join(graphs_root, os.path.basename(p)) if not os.path.isabs(p) else p)

    data_list = []
    use_pyg = to_pyg
    if to_pyg:
        try:
            from torch_geometric.data import Data
        except Exception:
            use_pyg = False

    for _, row in df.iterrows():
        gpath = row['graph_path']
        if not os.path.exists(gpath):
            print(f"Missing graph file: {gpath}, skipping")
            continue
        g = torch.load(gpath)
        x = g.get('node_attr')
        edge_index = g.get('edge_index')
        edge_attr = g.get('edge_attr') if 'edge_attr' in g else None
        yval = row.get(target_col, None)
        if pd.isna(yval):
            if replace_nan_with is None:
                y = torch.tensor(float('nan'))
            else:
                y = torch.tensor(float(replace_nan_with), dtype=torch.float)
        else:
            y = torch.tensor(float(yval), dtype=torch.float).unsqueeze(0)

        if use_pyg:
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.smiles = g.get('smiles')
            data.ligand_id = g.get('ligand_id')
            data_list.append(data)
        else:
            data_list.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'smiles': g.get('smiles'), 'ligand_id': g.get('ligand_id')})

    return data_list

# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_dir', type=str, required=True, help='Directory with batch*_score.txt files')
    parser.add_argument('--out_csv', type=str, default=None)
    parser.add_argument('--graphs_dir', type=str, default=None)
    parser.add_argument('--use_pos', action='store_true')
    args = parser.parse_args()

    build_graphs_from_batch_files(args.sdf_dir, out_csv_path=args.out_csv, graphs_dir=args.graphs_dir, use_pos=args.use_pos)

if __name__ == '__main__':
    main()
