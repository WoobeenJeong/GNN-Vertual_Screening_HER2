# GNN-Vertual_Screening_HER2
### Large-scale Virtual Screening using GNN 


This repository is part of the LAIDD mentoring project.
For internal and educational use only. Do not redistribute without permission.

---

The dataset is available at:

### [ Development set (100,160 compounds) ]

Hit Locator Library (HLL-100)

https://enamine.net/compound-libraries/diversity-libraries

Final Scored Ligand set ( 97,956 compounds )

### [ Evaluation set (4,643,787 compounds) ]

https://enamine.net/compound-collections/screening-collection

---

# Abstract

Building on the success of HER2-targeted therapies like tucatinib in breast cancer, the strategy is now being applied to other HER2-altered tumors.
HER2-mutant metastatic non-small cell lung cancer (mNCLC), notably in Asian female non-smokers, demonstrates only modest benefit from current targeted therapies (≈30–50% ORR; median targeted therapy duration ~8 months) and lacks an FDA-approved standard, while resistance driven by wild-type HER2 relapse and exon-20 insertion variants remains inadequately addressed.<br>
<br>
<img width="2000" height="745" alt="image" src="https://github.com/user-attachments/assets/8430658c-3306-421e-98d6-28e04659f535" /><br>
<br>
We performed large-scale computer-based virtual screening targeting the HER2 8VB5 binding region to identify and prioritize candidate compounds/proteins with predicted high affinity, improved potential activity against exon-20 insertion mutants.<br>
<br>
Top hits were triaged using deep-learning frameworks and are recommended for biochemical and cellular validation as promising leads to overcome current resistance mechanisms and improve clinical outcomes.<br>
<br>
<img width="1661" height="844" alt="image" src="https://github.com/user-attachments/assets/193bdc5d-c1d6-43ab-80ce-a1a4ccd0f6cc" /><br>
<br>

# Baselines

For reproducibility, we commit environment.yml

Workflow note: for AWS EC2 (x86_64)

```
# from a terminal where the environment exists
conda activate docking
conda env export --from-history --name docking > environment.yml
```
--from-history ensures only explicitly installed packages are listed.<br>
<br>
Project directory structure on AWS EC2 (To preprocessing data):
```
/home/ssm-user/project
├─ 01_check_batch.ipynb
├─ 02_autodock_batch.ipynb
├─ 03_make_smi_from_csv.py
├─ 04_prepare_pdbqt_parallel.sh
├─ 05_gnina.sh
├─ 06_make_gnina_list.py
├─ smiles_batch_1~5.csv                 ## Ligands
└─ autodock/
   ├─ *.cif, pdb, pdbqt, map, mol2      ## Target
   ├─ ligand_list.txt
   ├─ potent/
      └─ *.smi
   └─ potent_pdbqt/
      └─ *.pdbqt
├─ gnina/
      └─ *.sdf
└─ scores/
   ├─ 07_generalize_output.ipynb
   ├─ 08_smiles_to_graph.py
   ├─ own_docking_prediction_models.py
   ├─ input_graphs.csv                  ## Meta-data (n=97,956)    
   └─ graphs/
      └─*.pt
```

# Autodock-GPU (Vina affinity)

Fast version of docking tool.
- Physics-based prediction on small molecule (drug candidates) by their molecular interaction with target.
- GPU acceleration allows fast screening of large compound libraries, purely physical scoring may miss subtle patterns.

# GNINA (CNN affinity)

Modern version of docking tool.
- Extends docking by integrating a convolutional neural network (CNN) to score ligand poses.
- Improves pose ranking and virtual screening performance, especially for challenging targets, by capturing patterns AutoDock GPU alone may miss.

# ML & DL based TOP-n compound picking

<details>
  <summary>XGBoost    (Dohoon Kim)</summary>

   XGB : Uses gradient-boosted decision trees (GBDT) to iteratively correct errors from previous trees, optimizing predictive performance.

</details>

<details>
  <summary>LightGBM   (Dohoon Kim)</summary>

   LGBM : GBDT-based algorithm, optimized for speed and memory efficiency using leaf-wise tree growth instead of level-wise.
   
</details>

<details>
  <summary>Ensembled GNN   (Hyojin Kim, Woobeen Jeong)</summary>   

   GCN : Basic convolutional networks for graph structured input.
   
   GATv2 : Uses attention to weigh neighbors dynamically, To improve representation of feature relations.
   
   Transformer-GNN : Combine graph structured input with self-attention to capture long-range dependancies.
   
   GINEconv : Encode edge and node features together to create rich graph profiles.

</details>

- " " shows the best performance and interpretation on druggability

# Analyze interaction properties using ChimeraX


# Collaborators
### Professor
- Juyong Lee, Seoul National University
### Mentor
- Haelyn Kim, Seoul National University
### Contributors
Research scientist in the pharmaceutical industry

- Dohoon Kim
- Hyojin Kim

AI research scientist in the pharmaceutical industry

- Bokyung Park
- Woobeen Jeong

Korea Advanced Institute of Science and Technology (KAIST)

- Junseo Hwang
