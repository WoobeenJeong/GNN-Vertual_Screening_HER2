# save as build_graphs_from_batches.py and run in environment with RDKit + torch + pandas
import os
import glob
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------------
# Gnina-like convert function
# -------------------------

def convert_mol_to_graph_gnina_like(mol, use_pos=False):
    ## remove explicit H for graph simplicity (as original)
    mol2 = Chem.RemoveHs(mol)
    n_bonds = len(mol2.GetBonds())
    n_atoms = len(mol2.GetAtoms())

    edge_index = []
    edge_attr = []
    edge_weight = []

    ## edges
    for edge_idx in range(n_bonds):
        bond = mol2.GetBondWithIdx(edge_idx)
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edge_index.append([a, b])
        edge_index.append([b, a])

        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bond_one_hot = [1,0,0,0]; edge_weight.extend([1.0, 1.0])
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bond_one_hot = [0,1,0,0]; edge_weight.extend([1.5, 1.5])
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bond_one_hot = [0,0,1,0]; edge_weight.extend([2.0, 2.0])
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bond_one_hot = [0,0,0,1]; edge_weight.extend([3.0, 3.0])
        else:
            bond_one_hot = [0,0,0,0]; edge_weight.extend([1.0,1.0])

        stype = bond.GetStereo()
        stereo_one_hot = [0]*6
        if stype == Chem.rdchem.BondStereo.STEREOANY:
            stereo_one_hot[0]=1
        elif stype == Chem.rdchem.BondStereo.STEREOCIS:
            stereo_one_hot[1]=1
        elif stype == Chem.rdchem.BondStereo.STEREOE:
            stereo_one_hot[2]=1
        elif stype == Chem.rdchem.BondStereo.STEREONONE:
            stereo_one_hot[3]=1
        elif stype == Chem.rdchem.BondStereo.STEREOTRANS:
            stereo_one_hot[4]=1
        elif stype == Chem.rdchem.BondStereo.STEREOZ:
            stereo_one_hot[5]=1

        ring_bond = 1 if bond.IsInRing() else 0
        conjugate = 1 if bond.GetIsConjugated() else 0

        attr = bond_one_hot + stereo_one_hot + [ring_bond, conjugate]
        edge_attr.append(attr)
        edge_attr.append(attr)

    ## node features (Gnina-like Appended to metals and halogens)
    valid_atoms = {
      'H':0,'B':1,'C':2,'N':3,'O':4,'F':5,'P':6,'S':7,'Cl':8,
      'Br':9,'I':10,'Fe':11,'Zn':12,'Mg':13,'Ca':14,'Na':15,'K':16,'OTHER':17
    }
    metals = {'Fe','Zn','Mg','Ca','Na','K','Cu','Mn','Co','Ni'}
    halogens = {'F','Cl','Br','I'}

    ## compute Gasteiger charges (silent fail safe)
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
        idx = valid_atoms.get(sym, valid_atoms['OTHER'])
        atm_one_hot[idx] = 1

        hybrid = atm.GetHybridization()
        hybrid_one_hot = [0]*7
        if hybrid == Chem.HybridizationType.SP3: hybrid_one_hot[0]=1
        elif hybrid == Chem.HybridizationType.SP2: hybrid_one_hot[1]=1
        elif hybrid == Chem.HybridizationType.SP:  hybrid_one_hot[2]=1
        elif hybrid == Chem.HybridizationType.S:   hybrid_one_hot[3]=1
        elif hybrid == Chem.HybridizationType.SP3D: hybrid_one_hot[4]=1
        elif hybrid == Chem.HybridizationType.SP3D2: hybrid_one_hot[5]=1
        else: hybrid_one_hot[6]=1

        arom = 1 if atm.GetIsAromatic() else 0
        ring_flag = 1 if atm.IsInRing() else 0

        degree_one_hot = [0]*6
        degree = atm.GetTotalDegree()
        degree_one_hot[5 if degree>=5 else degree] = 1

        num_h = atm.GetTotalNumHs()
        hydrogen_one_hot = [0]*5
        hydrogen_one_hot[4 if num_h>=4 else num_h] = 1

        chiral = atm.GetChiralTag()
        if chiral == Chem.rdchem.ChiralType.CHI_OTHER:
            chiral_one_hot = [1,0,0,0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral_one_hot = [0,1,0,0]
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            chiral_one_hot = [0,0,1,0]
        else:
            chiral_one_hot = [0,0,0,1]

        ## additional features
        sym_upper = sym
        is_hbd = 0
        if sym_upper in {'N','O','S'}:
            if atm.GetTotalNumHs() > 0:
                is_hbd = 1
        is_hba = 1 if sym_upper in {'N','O','F','S','P','Cl'} else 0
        is_halogen = 1 if sym_upper in halogens else 0
        is_metal = 1 if sym_upper in metals else 0

        neighbor_polar = 0
        for neigh in atm.GetNeighbors():
            if neigh.GetSymbol() in {'N','O','F','Cl','S','P'}:
                neighbor_polar = 1
                break

        g_charge = 0.0
        if has_gasteiger:
            try:
                g_charge = float(atm.GetProp('_GasteigerCharge'))
            except Exception:
                g_charge = 0.0

        atomic_num = atm.GetAtomicNum()
        atomic_mass = atm.GetMass()

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
        pos_list = []
        conf = mol2.GetConformer(0)
        for atm_id in range(n_atoms):
            p = conf.GetAtomPosition(atm_id)
            pos_list.append([p.x, p.y, p.z])
        pos = torch.tensor(pos_list, dtype=torch.float)
    else:
        pos = None

    return edge_index, node_attr, edge_attr, pos, edge_weight

# -------------------------
# Main processing pipeline
# -------------------------

def build_graphs_from_batch_files(SDF_DIR, out_csv_path=None, graphs_dir=None, use_pos=False):
    if out_csv_path is None:
        out_csv_path = os.path.join(SDF_DIR, "processed_graphs.csv")
    if graphs_dir is None:
        graphs_dir = os.path.join(SDF_DIR, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    file_pattern = os.path.join(SDF_DIR, "batch*_score.txt")
    files = sorted(glob.glob(file_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

    rows_out = []
    total = 0
    kept = 0
    for fpath in files:
        batch_name = os.path.basename(fpath)
        print(f"Processing {batch_name} ...")
        ## try pandas auto-detect separator
        try:
            df = pd.read_csv(fpath, sep=None, engine='python')
        except Exception:
            # fallback: tab separated
            df = pd.read_csv(fpath, sep='\t', engine='python')

        ## Normalize column names to access needed fields robustly
        columns_lower = {c.lower(): c for c in df.columns}
        
        ## find SMILES col
        smiles_col = None
        for candidate in ['smiles', 'smile', 'SMILES']:
            if candidate in columns_lower:
                smiles_col = columns_lower[candidate]
                break
        if smiles_col is None:
            # try fuzzy match
            for c in df.columns:
                if 'smiles' in c.lower():
                    smiles_col = c
                    break
        if smiles_col is None:
            raise ValueError(f"Could not find SMILES column in {fpath}. Columns: {df.columns.tolist()}")

        ## find vina affinity col
        ba_col = None
        for cand in ['binding affinity (kcal/mol)', 'Binding affinity (kcal/mol)', 'vina_affinity']:
            if cand in columns_lower:
                ba_col = columns_lower[cand]
                break

        ## find gnina affinity col
        cnn_col = None
        for cand in ['cnn affinity', 'CNN affinity', 'gnina_affinity']:
            if cand in columns_lower:
                cnn_col = columns_lower[cand]
                break
        if ba_col is None:
            for c in df.columns:
                if 'binding' in c.lower() and 'kcal' in c.lower():
                    ba_col = c; break
        if cnn_col is None:
            for c in df.columns:
                if 'cnn' in c.lower() and 'aff' in c.lower():
                    cnn_col = c; break

        ## iterate rows
        for idx, row in df.iterrows():
            total += 1
            smiles = str(row[smiles_col]).strip()
            ligand_id = row.get('Ligand', row.get('ligand', f"{os.path.splitext(batch_name)[0]}_{idx}"))
            # fetch ba and cnn if available
            ba_val = row.get(ba_col, None) if ba_col is not None else None
            cnn_val = row.get(cnn_col, None) if cnn_col is not None else None

            ## Attempt to parse SMILES robustly
            mol = None
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # try sanitize=False then sanitize
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception:
                            # try to force sanitize by removing problematic atoms? skip for safety
                            mol = None
            except Exception:
                mol = None

            if mol is None:
                ## last-resort: try kekulize/smiles roundtrip
                try:
                    tmp = Chem.MolFromSmiles(smiles, sanitize=False)
                    if tmp is not None:
                        Chem.SanitizeMol(tmp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                        mol = tmp
                except Exception:
                    mol = None

            if mol is None:
                continue

            ## Convert to graph
            try:
                res = convert_mol_to_graph_gnina_like(mol, use_pos=use_pos)
                if res is None:
                    continue
                edge_index, node_attr, edge_attr, pos, edge_weight = res
            except Exception as e:
                # conversion failed -> skip
                print(f"Conversion error for {ligand_id} (idx {idx}) : {e}")
                continue

            ## save graph object to file
            safe_name = f"{os.path.splitext(batch_name)[0]}_{idx}_{str(ligand_id)}".replace(os.sep, "_").replace(" ", "_")
            graph_path = os.path.join(graphs_dir, safe_name + ".pt")
            # ensure deterministic small object by converting tensors to CPU
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

# -------------------------
# 실행 예시
# -------------------------

if __name__ == "__main__":
    SDF_DIR = "/home/ssm-user/project/scores"
    out_csv = os.path.join(SDF_DIR, "input_graphs.csv")
    graphs_dir = os.path.join(SDF_DIR, "graphs")
    df_meta = build_graphs_from_batch_files(SDF_DIR, out_csv_path=out_csv, graphs_dir=graphs_dir, use_pos=False)
    print(df_meta.head())
