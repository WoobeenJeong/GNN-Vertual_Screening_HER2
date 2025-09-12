#!/usr/bin/env python3
"""
Command Example:
    python 08_new_smiles_to_graph.py

Created Features:

[ 12 Edges ]

    1. bond type (Single, Aromatic, Double, Triple) = 4
    2. stereo type (Cis, Trans, E/Z, ...)           = 6
    3. ring_bond
    4. conjugate

[ 111 Nodes ]

    1. atom type (H, O, ..., Mg, ...)       = 18
    2. hybrid type (sp2, sp3, ...)          = 7
    3. degree (1, 2, 3, ...)                = 6
    4. # of H (1, 2, 3, ...)                = 5
    5. chirality (cw, ccw, ...)             = 4
    6. structural (arom, ring, ...)         = 4
    7. interaction (hbd, hba, halogen, ...) = 5

    8. gasteiger charge (Gaussian Embeded)  = 11
    9. atomic properties (Gaussian Embeded) = 30
    10. atom mass (Gaussian Embeded)        = 21

[ 3D structure ]

    1. solvent accessible surface area      = 4
    2. molecular volume                     = 4
    3. radius of gyration                   = 4

    4. HER2 TKI specific hindge motif       = 4
    5. Cys805 Covalent Warhead
    6. Met801, Cys805, Asp863, Ser783 interaction
    
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# --------------
# 3D feature
# --------------

def get_global_features_from_mol(mol, num_confs=10):
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params)

        if len(cids) == 0:
            if AllChem.EmbedMolecule(mol_3d, params=params) == -1: return None
            cids = [0]
        
        res = AllChem.MMFFOptimizeMoleculeConfs(mol_3d, mmffVariant='MMFF94s')
        converged_confs = [cid for cid, (conv, energy) in zip(cids, res) if conv == 0]
        if not converged_confs: return None

        energies = [item[1] for item in res if item[0] in converged_confs]
        if not energies: return None
        min_energy_cid = converged_confs[np.argmin(energies)]
        conf = mol_3d.GetConformer(min_energy_cid)
        
        sasa_list, vol_list, gyr_list = [], [], []
        for cid in converged_confs:
            sasa_list.append(rdMolDescriptors.CalcSASA(mol_3d, confId=cid))
            vol_list.append(AllChem.GetConformerVolume(mol_3d, confId=cid))
            gyr_list.append(rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=cid))
        
        if not sasa_list: return None

        d3_descriptors = [
            np.mean(sasa_list), np.std(sasa_list), np.min(sasa_list), np.max(sasa_list),
            np.mean(vol_list), np.std(vol_list), np.min(vol_list), np.max(vol_list),
            np.mean(gyr_list), np.std(gyr_list), np.min(gyr_list), np.max(gyr_list),
        ]
    except Exception as e:
        print(f"  [DEBUG] An error occurred in 3D generation part: {e}")
        import traceback
        traceback.print_exc()
        return None

    ## Hindge binding motifs (0/1) ##
    
    hinge_patterns = {
        'quinazoline': Chem.MolFromSmarts('c12ccccc1nc(N)cn2'),
        'pyrimidine': Chem.MolFromSmarts('c1c(N)ccnc1N'),
        'purine': Chem.MolFromSmarts('c12c(nc(N)nc1)ncn2'),
        'pyrrolopyrimidine': Chem.MolFromSmarts('c1c2c(nc(N)n1)cncn2')
    }
    hinge_features = [1.0 if mol.HasSubstructMatch(p) else 0.0 for p in hinge_patterns.values()]

    ## Warehead ##

    acrylamide_pattern = Chem.MolFromSmarts('C=CC(=O)N')
    has_covalent_warhead = 1.0 if mol.HasSubstructMatch(acrylamide_pattern) else 0.0

    ## Backbone interaction ##
    
    donor_coords = [conf.GetAtomPosition(idx) for idx in mol_3d.GetSubstructMatches(h_bond_donors)[0]] if mol_3d.HasSubstructMatch(h_bond_donors) else []
    acceptor_coords = [conf.GetAtomPosition(idx) for idx in mol_3d.GetSubstructMatches(h_bond_acceptors)[0]] if mol_3d.HasSubstructMatch(h_bond_acceptors) else []

    mock_positions = {
        'Met801_backbone': np.array([-1.0, 7.0, -16.0]),
        'Cys805_sulfur': np.array([-3.0, 4.0, -19.0]),
        'Asp863_carboxyl': np.array([2.0, 9.0, -21.0]),
        'Ser783_hydroxyl': np.array([1.5, 5.0, -15.0])
    }

    min_dist_met801 = min([np.linalg.norm(np.array(pos) - mock_positions['Met801_backbone']) for pos in acceptor_coords + donor_coords]) if acceptor_coords or donor_coords else 99.9
    min_dist_cys805 = min([np.linalg.norm(np.array(pos) - mock_positions['Cys805_sulfur']) for pos in conf.GetPositions()])
    min_dist_asp863 = min([np.linalg.norm(np.array(pos) - mock_positions['Asp863_carboxyl']) for pos in donor_coords]) if donor_coords else 99.9
    min_dist_ser783 = min([np.linalg.norm(np.array(pos) - mock_positions['Ser783_hydroxyl']) for pos in acceptor_coords]) if acceptor_coords else 99.9

    interaction_features = [
        1.0 if min_dist_met801 < 3.5 else 0.0, 
        1.0 if min_dist_cys805 < 2.5 else 0.0, 
        1.0 if min_dist_asp863 < 3.5 else 0.0, 
        1.0 if min_dist_ser783 < 3.5 else 0.0, 
    ]  

    global_attr = torch.tensor(d3_descriptors + hinge_features + [has_covalent_warhead] + interaction_features, dtype=torch.float)
    
    return global_attr

# --------------
# SMILES PARSE
# --------------

def parse_smiles_with_detailed_logging(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return mol, "Success: Standard parsing with sanitization."
    except Exception as e:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol)
            return mol, "Success: Parsed with sanitize=False, then sanitized."
    except Exception as e:
        return None, f"Failed at SanitizeMol step. Reason: {e}"
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol, "Success: Sanitized while skipping property calculation."
    except Exception as e:
        return None, f"Failed at SanitizeMol (skipping properties) step. Reason: {e}"
    return None, "All parsing attempts failed. The SMILES string might be fundamentally invalid."

# --------------
# Node and Edge
# --------------

def _is_rotatable_bond(bond, mol):
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE: return False
    if bond.IsInRing(): return False
    a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
    if a1.GetDegree() == 1 or a2.GetDegree() == 1: return False
    return True

def gaussian_embedding(value, centers, sigma=1.0):
    return [np.exp(-0.5 * ((value - c) / sigma) ** 2) for c in centers]

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

        ## Add Gausian Embedded Charge Info ##
        
        gcharge_emb = gaussian_embedding(g_charge, centers=np.linspace(-1,1,11), sigma=0.2)
        anum_emb    = gaussian_embedding(float(atomic_num), centers=np.arange(1, 31), sigma=1.0)
        amass_emb   = gaussian_embedding(float(atomic_mass), centers=np.linspace(0, 200, 21), sigma=5.0)
        
        attr += gcharge_emb
        attr += anum_emb
        attr += amass_emb

        node_attr.append(attr)
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if len(edge_attr)>0 else torch.empty((0,))
    node_attr = torch.tensor(node_attr, dtype=torch.float) if len(node_attr)>0 else torch.empty((0,))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if len(edge_index)>0 else torch.empty((2,0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float) if len(edge_weight)>0 else torch.empty((0,))

    if use_pos:
        try:
            val = AllChem.EmbedMolecule(mol2)
            if val != 0:
                print(f"Error while generating 3D for: {Chem.MolToSmiles(mol)}")
                return None
            pos_list = []; conf = mol2.GetConformer(0)
            for atm_id in range(n_atoms):
                p = conf.GetAtomPosition(atm_id); pos_list.append([p.x,p.y,p.z])
            pos = torch.tensor(pos_list, dtype=torch.float)
        except Exception as e:
            print(f"[warn] 3D embedding failed: {e}")
            pos = None
    else:
        pos = None

    return edge_index, node_attr, edge_attr, pos, edge_weight

# --------------
# Build Graph
# --------------
def build_graphs_from_batch_files(SDF_DIR, out_csv_path=None, graphs_dir=None, use_pos=False, args=None):
    if out_csv_path is None: out_csv_path = os.path.join(SDF_DIR, "processed_graphs_3d.csv")
    if graphs_dir is None: graphs_dir = os.path.join(SDF_DIR, "graphs_3d")
    os.makedirs(graphs_dir, exist_ok=True)

    file_pattern = os.path.join(SDF_DIR, "batch*_score.txt")
    files = sorted(glob.glob(file_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

    rows_out = []
    total = 0; kept = 0
    for fpath in files:
        batch_name = os.path.basename(fpath)
        print(f"Processing {batch_name} ...")
        try:
            df = pd.read_csv(fpath, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(fpath, sep='\t', engine='python')
        columns_lower = {c.lower(): c for c in df.columns}

        ## GET SMILES ##
        
        smiles_col = None
        for candidate in ['smiles', 'smile', 'smilestring']:
            if candidate in columns_lower:
                smiles_col = columns_lower[candidate]; break
        if smiles_col is None:
            for c in df.columns:
                if 'smiles' in c.lower():
                    smiles_col = c; break
        if smiles_col is None:
            raise ValueError(f"Could not find SMILES column in {fpath}. Columns: {df.columns.tolist()}")

        ## Get VINA ##
        
        vina_col = None
        vina_candidates = ['vina_affinity','vina affinity','vina','binding_affinity_kcal_mol',
                           'binding affinity (kcal/mol)','binding affinity','binding_affinity','affinity','score']
        for cand in vina_candidates:
            if cand in columns_lower:
                vina_col = columns_lower[cand]; break
        if vina_col is None:
            for c in df.columns:
                lc = c.lower()
                if 'vina' in lc or ('binding' in lc and 'kcal' in lc) or ('affin' in lc and 'cnn' not in lc):
                    vina_col = c; break

        ## Get GNINA ##
        
        gnina_col = None
        gnina_candidates = ['gnina_affinity','gnina affinity','gnina','cnnaffinity','cnn affinity','cnscore','cnnaff','cnn_affinity']
        for cand in gnina_candidates:
            if cand in columns_lower:
                gnina_col = columns_lower[cand]; break
        if gnina_col is None:
            for c in df.columns:
                lc = c.lower()
                if 'gnina' in lc or ('cnn' in lc and 'aff' in lc) or 'cnn' in lc:
                    gnina_col = c; break
        print(f"  Detected columns for file {batch_name}: smiles='{smiles_col}', vina='{vina_col}', gnina/cnn='{gnina_col}'")

        ## SMILES Parsing for 3D 
        
        for idx, row in df.iterrows():
            total += 1
            smiles = str(row[smiles_col]).strip()
            ligand_id = row.get('Ligand', row.get('ligand', row.get('ligand_id', f"{os.path.splitext(batch_name)[0]}_{idx}")))
            ba_val_raw = row.get(vina_col, None) if vina_col is not None else None
            cnn_val_raw = row.get(gnina_col, None) if gnina_col is not None else None

            mol, log_message = parse_smiles_with_detailed_logging(smiles)
            
            if mol is None:
                print(f"  [skip] cannot parse SMILES for ligand_id={ligand_id} smiles='{smiles[:50]}'")
                print(f"  [DEBUG] Reason: {log_message}")
                continue

            ## 3D structure ##
            
            global_attr = get_global_features_from_mol(mol, num_confs=args.num_confs)
            if global_attr is None:
                print(f"  [skip] Failed to generate 3D/Hinge features for {ligand_id}")
                continue

            ## Graph Component ## 
            
            try:
                res = convert_mol_to_graph_gnina_like(mol, use_pos=use_pos)
                if res is None:
                    print(f"  [skip] conversion returned None for ligand_id={ligand_id}")
                    continue
                edge_index, node_attr, edge_attr, pos, edge_weight = res
            except Exception as e:
                print(f"  [skip] conversion error for ligand_id={ligand_id} idx={idx} : {e}")
                continue
            
            ## Save ##
            
            safe_name = f"{os.path.splitext(batch_name)[0]}_{idx}_{str(ligand_id)}".replace(os.sep, "_").replace(" ", "_")
            graph_path = os.path.join(graphs_dir, safe_name + ".pt")
            graph_obj = {
                "edge_index": edge_index.cpu() if isinstance(edge_index, torch.Tensor) else edge_index,
                "node_attr": node_attr.cpu() if isinstance(node_attr, torch.Tensor) else node_attr,
                "edge_attr": edge_attr.cpu() if isinstance(edge_attr, torch.Tensor) else edge_attr,
                "global_attr" : global_attr.cpu() if isinstance(global_attr, torch.Tensor) else global_attr,
                "pos": pos.cpu() if isinstance(pos, torch.Tensor) else pos,
                "edge_weight": edge_weight.cpu() if isinstance(edge_weight, torch.Tensor) else edge_weight,
                "smiles": smiles,
                "ligand_id": ligand_id
            }
            try:
                torch.save(graph_obj, graph_path)
            except Exception as e:
                print(f"  [warn] failed to save graph file {graph_path}: {e}")
                continue

            # robust numeric conversion for affinities
            try:
                ba_val = float(pd.to_numeric(ba_val_raw, errors='coerce')) if ba_val_raw is not None else None
            except Exception:
                ba_val = None
            try:
                cnn_val = float(pd.to_numeric(cnn_val_raw, errors='coerce')) if cnn_val_raw is not None else None
            except Exception:
                cnn_val = None

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
    return out_df

# --------------
# Helper
# --------------

def load_graphs_for_gnn(meta_csv, graphs_root=None, to_pyg=True, replace_nan_with=None):
    """
    Loads saved graph .pt objects and returns:
      - if to_pyg and torch_geometric is available: list of torch_geometric.data.Data objects
      - else: list of dicts with keys: x (node_attr tensor), edge_index (tensor), edge_attr (tensor or None), y (tensor)
    The meta CSV is expected to have columns: graph_path, smiles, ligand_id, vina_affinity, gnina_affinity
    replace_nan_with: float to replace NaN target values (optional)
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

        # gather two targets (vina, gnina) into y tensor (shape [2])
        vina_val = row.get('vina_affinity', None)
        gnina_val = row.get('gnina_affinity', None)
        try:
            vina_val = float(pd.to_numeric(vina_val, errors='coerce')) if (vina_val is not None and not (isinstance(vina_val,float) and np.isnan(vina_val))) else (replace_nan_with if replace_nan_with is not None else float('nan'))
        except Exception:
            vina_val = replace_nan_with if replace_nan_with is not None else float('nan')
        try:
            gnina_val = float(pd.to_numeric(gnina_val, errors='coerce')) if (gnina_val is not None and not (isinstance(gnina_val,float) and np.isnan(gnina_val))) else (replace_nan_with if replace_nan_with is not None else float('nan'))
        except Exception:
            gnina_val = replace_nan_with if replace_nan_with is not None else float('nan')

        y = torch.tensor([gnina_val, vina_val], dtype=torch.float)  # note order: [gnina, vina] if you prefer change accordingly

        if use_pyg:
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.smiles = g.get('smiles')
            data.ligand_id = g.get('ligand_id')
            data_list.append(data)
        else:
            data_list.append({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'y': y, 'smiles': g.get('smiles'), 'ligand_id': g.get('ligand_id')})

    return data_list

# --------------
# Main
# --------------

def main():
    parser = argparse.ArgumentParser(description="Build graphs from batch*_score.txt and write metadata CSV")
    parser.add_argument('--sdf_dir', type=str, default='/home/ssm-user/project/scores', help='Directory with batch*_score.txt files')
    parser.add_argument('--out_csv', type=str, default=None, help='where to write processed_graphs.csv')
    parser.add_argument('--graphs_dir', type=str, default=None, help='directory to save .pt graph files')
    parser.add_argument('--use_pos', action='store_true', help='attempt to generate 3D coords (slow)')
    parser.add_argument('--num_confs', type=int, default=10, help='Number of conformers to generate per molecule.')
    args = parser.parse_args()

    build_graphs_from_batch_files(args.sdf_dir, out_csv_path=args.out_csv, graphs_dir=args.graphs_dir, use_pos=args.use_pos, args=args)

if __name__ == '__main__':
    main()
