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

1. MedChem Filtered Dataset

2. Immediate SAR from Clustered Set


### [ Screening set (4,643,787 compounds) ]

https://enamine.net/compound-collections/screening-collection

Final Result Ligand set ( 3,188,651 compounds )

1. The world’s largest screening collection

2. Available as pre-plated sets


---

# Abstract

<img width="2283" height="578" alt="image" src="https://github.com/user-attachments/assets/1ea9d044-cae4-4786-9753-300668229702" />


Building on the success of HER2-targeted therapies like tucatinib, the strategy is now being applied to other HER2-altered tumors.

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
Project directory structure on AWS EC2 (To preprocess & build Bayes_GNE model)
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

# Graph Neural Network based TOP-n compound picking


<img width="1731" height="661" alt="image" src="https://github.com/user-attachments/assets/45062d21-5c23-4ad4-9c9b-dfa33d7c8477" />


<img width="2033" height="430" alt="image" src="https://github.com/user-attachments/assets/048be83e-06ba-4692-a005-d2e8931cedd6" />


<details>
  <summary>Ensembled GNN </summary>   

   GCN : Basic convolutional networks for graph structured input.
   
   GATv2 : Uses attention to weigh neighbors dynamically, To improve representation of feature relations.
   
   Transformer-GNN : Combine graph structured input with self-attention to capture long-range dependancies.
   
   GINEconv : Encode edge and node features together to create rich graph profiles.

</details>

```
(docking) $ python 002_bayes_GNE_inference.py

[info] torch_geometric's GNN layers are available.
[info] Using device: cuda
[info] Output will be saved to: /home/ssm-user/project/scores/results/enamine_ligands_predictions.csv

[info] Loaded 3188651 graphs for prediction.
[info] Inferred input feature dim: 120, Edge feature dim: 12
[info] Loading pre-trained ensemble models...
  > Loading 'gine' from /home/ssm-user/project/scores/results/ensemble_member_0_gine_best.pt
  > Loading 'gatv2' from /home/ssm-user/project/scores/results/ensemble_member_1_gatv2_best.pt
  > Loading 'transformer' from /home/ssm-user/project/scores/results/ensemble_member_2_transformer_best.pt
  > Loading 'gcn' from /home/ssm-user/project/scores/results/ensemble_member_3_gcn_best.pt
[info] Successfully loaded 4 models.

[info] Starting prediction with ensemble...
  > Sampling from model 1/4 (GINE) (MC-T=5)
  > Sampling from model 2/4 (GATV2) (MC-T=5)
  > Sampling from model 3/4 (TRANSFORMER) (MC-T=5)
  > Sampling from model 4/4 (GCN) (MC-T=5)
> 
[info] Prediction complete. Generated 20 samples for 3188651 ligands.
[info] Calculating final scores and confidence metrics...

-----------------[ Summary ]---------------------
Successfully predicted ligands: 3188651
Total inference time: 2232063.72 seconds (0.01 per ligand)
Rank confidence calculation time: 19.3441 seconds
---------------------------------------------------

<bash>

(head -n 1 enamine_ligands_predictions.csv && tail -n +2  enamine_ligands_predictions.csv \
   | sort -t, -k5,5nr -k8,8nr) | head -n 10001 > top10000_enamine_ligands_predictions_out.csv

```

<img width="2101" height="672" alt="image" src="https://github.com/user-attachments/assets/382aca7c-b918-4e3a-af67-bc112b6e925e" />


# Analyze Druggability properties


<img width="2244" height="669" alt="image" src="https://github.com/user-attachments/assets/cc0ad039-c64a-446d-b996-b7eee6306e0d" />


<img width="2201" height="883" alt="image" src="https://github.com/user-attachments/assets/5f24df75-5931-457e-a87e-25fe2335c135" />


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
