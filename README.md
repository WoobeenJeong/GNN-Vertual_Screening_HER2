## Bayes-GNE: A Bayesian-Approximated GNN Ensemble for HER2 Ligand discovery
### Large-scale Virtual Screening using GNN 

---

The dataset is available at :

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

## Abstract

<img width="2283" height="578" alt="image" src="https://github.com/user-attachments/assets/1ea9d044-cae4-4786-9753-300668229702" />


Building on the success of HER2-targeted therapies like tucatinib, the strategy is now being applied to other HER2-altered tumors.

We performed large-scale computer-based virtual screening targeting the HER2 8VB5 binding region to identify and prioritize candidate compounds/proteins with predicted high affinity, improved potential activity against exon-20 insertion mutants.<br>
<br>
Top hits were triaged using deep-learning frameworks and are recommended for biochemical and cellular validation as promising leads to overcome current resistance mechanisms and improve clinical outcomes.<br>

## Baselines

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

## Autodock-GPU (Vina affinity)

Fast version of docking tool.
- Physics-based prediction on small molecule (drug candidates) by their molecular interaction with target.
- GPU acceleration allows fast screening of large compound libraries, purely physical scoring may miss subtle patterns.

## GNINA (CNN affinity)

Modern version of docking tool.
- Extends docking by integrating a convolutional neural network (CNN) to score ligand poses.
- Improves pose ranking and virtual screening performance, especially for challenging targets, by capturing patterns AutoDock GPU alone may miss.

## TOP-n compound picking by using GNN based model (Bayes-GNE)


<img width="1731" height="661" alt="image" src="https://github.com/user-attachments/assets/45062d21-5c23-4ad4-9c9b-dfa33d7c8477" />


<img width="2033" height="430" alt="image" src="https://github.com/user-attachments/assets/048be83e-06ba-4692-a005-d2e8931cedd6" />


### Ensembled GNN

- **GCN** : Basic convolutional networks for graph structured input.

- **GATv2** : Uses attention to weigh neighbors dynamically, To improve representation of feature relations.

- **Transformer-GNN** : Combine graph structured input with self-attention to capture long-range dependancies.

- **GINE** : Encode edge and node features together to create rich graph profiles.

### Activation function : Sigmoid Linear Unit 

- SiLU(swish) provides smoother and self-gated nonlinearity than ReLU, helping GNN models capture subtle node interactions.

   ![SiLU](https://latex.codecogs.com/svg.image?\mathrm{SiLU}=x/(1+e^{-x})=\mathrm{swish}_{\beta=1})

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
Total inference time: 2232063.72 seconds (0.70 per ligand)
Rank confidence calculation time: 19.3441 seconds
---------------------------------------------------

<bash>

(head -n 1 enamine_ligands_predictions.csv && tail -n +2  enamine_ligands_predictions.csv \
   | sort -t, -k5,5nr -k8,8nr) | head -n 10001 > top10000_enamine_ligands_predictions_out.csv

```

<img width="2101" height="672" alt="image" src="https://github.com/user-attachments/assets/382aca7c-b918-4e3a-af67-bc112b6e925e" />

We designed Bayes-GNE (Bayesian-approximated GNN Ensemble) to mitigate two pervasive issues in structure-based virtual screening: 

(i) inconsistent and often discordant ranking signals from physics-based docking scores (e.g., AutoDock Vina) and pose-sensitive CNN scorers (e.g., GNINA), and (ii) the high computational cost of exhaustive docking and CNN rescoring. 

Each molecule is encoded as a graph of atom (node) and bond (edge) features and evaluated by an ensemble of four complementary GNN architectures; applying Monte-Carlo dropout (p = 0.2) at inference approximates Bayesian model averaging **to produce both affinity predictions (for Vina and GNINA targets) and posterior uncertainty estimates.** 

We define **rank confidence** as the probability that a molecule’s rank remains unchanged given model uncertainty. Using this measure, we obtain an uncertainty-aware consensus between docking and CNN predictions that focuses experimental follow-up and reduces the need for costly re-scoring.


## Analyze Druggability properties

- **QED** (Quantitative Estimate of Drug-likeness) : a single 0–1 score that estimates how drug-like a molecule is; we used QED ≥ 0.7 as the threshold.

- **SA** (Synthetic Accessibility) : a heuristic score estimating how difficult a molecule is to synthesize (lower = easier); we used SA < 3.0 as the threshold.

From the top 10,000 ranked compounds, PAINS filtering (pan-assay interference compound removal) together with the QED and SA thresholds reduced the set to 3,353 ligands. 

Of the ligands remaining after PAINS filtering (≈34% of the original top 10,000), we ranked candidates using rank confidence, a Bayes-GNE-specific score, and selected the top 10. 

We then checked these ten for the canonical hinge-binding motif typical of HER2 tyrosine kinase inhibitors; 6 of the 10 (60%) contained the motif. Finally, using Boltz for visualization and scoring, we reviewed these molecules and chose 5 final candidate ligands.

<img width="2244" height="669" alt="image" src="https://github.com/user-attachments/assets/cc0ad039-c64a-446d-b996-b7eee6306e0d" />


<img width="2201" height="883" alt="image" src="https://github.com/user-attachments/assets/5f24df75-5931-457e-a87e-25fe2335c135" />


## CRediT statement

**Professor (Seoul National University)**
- Juyong Lee : Supervision; Conceptualization; Methodology; Funding acquisition.

  (Research supervision; proposed Gaussian embedding for node/edge features.)

**Mentor (Seoul National University)**
- Haelyn Kim : Resources; Project administration; Methodology; Supervision.

  (Built and maintained the analysis environment; advised on data interpretation and overall project management.)
  
**Research scientist in the pharmaceutical industry**
- Dohoon Kim : Data curation; Software; Validation; Formal analysis.
  
  (Data splitting and preprocessing; initial LGBM/GNN checks and troubleshooting; Boltz affinity verification.)

- Hyojin Kim : Conceptualization; Methodology; Validation; Investigation.

  (Proposed adding hinge-binding as graph edges; suggested ensemble strategy and hinge-based ligand selection.)

- Bokyung Park : Conceptualization; Data curation; Validation; Investigation.

  (Recommended HER2 target 8VB5 and using its affinity as reference; applied PAINS filtering to reduce candidate set.)

**Seoul National University**
- Woobeen Jeong : Conceptualization; Methodology; Formal analysis; Software.

  (Designed GNN-ensemble statistical framework and proposed the Bayesian-approximation with rank confidence.)

**Korea Advanced Institute of Science and Technology (KAIST)**
- Junseo Hwang : Conceptualization; Methodology; Formal analysis; Software.

  (Analyzed inter-affinity discrepancies; advocated rank-focused prioritization; defined model performance metrics.)

## License

This repository is licensed under the  
[Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

© 2025 LAIDD Mentoring Project. For educational and internal research use only.
