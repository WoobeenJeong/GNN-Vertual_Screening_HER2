# GNN-Vertual_Screening_HER2
### Large-scale Virtual Screening using GNN 


This repository is part of the LAIDD mentoring project.
For internal and educational use only. Do not redistribute without permission.

---

# Timeline for collaborators (removed after completion)

### [ 선정타겟 PDB ]

8VB5 (특징: chain A / YVMA mutation)

### [ 특이사항 ]

8/13
- HETATM 제거: EDO, PEG (Non-standard residue)
- Native legand pdbqt : A1AAC (이미 tucatinib 붙는곳, 논문은 아직X)

8/14
- 10만개 리스트 5 split 분할

8/16
- 5 split 중, 2번 split에서 중복값 3개 제외
- split별 통계적 유의성 검토

8/17
- maxit (RCSB) cif -> pdb 변환

### [ 목표 ]

*활동일지=줌캡쳐+요약

8/20 = Autodock으로 도킹 완료(affinity, smiles 따오기)

9/3 = 모델 완성 및 Top 뽑아서 새로 도킹(GININA로 score 뽑을 여유 있으면 수행)

9/10 = Boltz structure 구성

9/14 = 포스터 구성 완성


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
Project directory structure on AWS EC2:
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
   ├─ input_graphs.csv                  ## Meta-data (n=97,956)    
   └─ graphs/
      └─*.pt
```

# Autodock-GPU

I will describe it later on...

# GNINA


# ML & DL based TOP-n compound picking


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
