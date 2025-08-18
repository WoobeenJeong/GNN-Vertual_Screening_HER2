# GNN-Vertual_Screening_HER2
### Large-scale Virtual Screening (VS) using Docking and GNN 


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

The dataset is available at

https://enamine.net/compound-libraries/diversity-libraries

---

# Abstract

HER2-mutant metastatic non-small cell lung cancer (mNCLC), notably in Asian female non-smokers, demonstrates only modest benefit from current targeted therapies (≈30–50% ORR; median targeted therapy duration ~8 months) and lacks an FDA-approved standard, while resistance driven by wild-type HER2 relapse and exon-20 insertion variants remains inadequately addressed.

---

<img width="2000" height="745" alt="image" src="https://github.com/user-attachments/assets/8430658c-3306-421e-98d6-28e04659f535" />

---

We performed large-scale computer-based virtual screening targeting the HER2 8VB5 binding region to identify and prioritize candidate compounds/proteins with predicted high affinity, improved potential activity against exon-20 insertion mutants.

Top hits were triaged using deep-learning frameworks and are recommended for biochemical and cellular validation as promising leads to overcome current resistance mechanisms and improve clinical outcomes.

---

<img width="1661" height="844" alt="image" src="https://github.com/user-attachments/assets/193bdc5d-c1d6-43ab-80ce-a1a4ccd0f6cc" />

---

# Baselines

I will describe it later on...

# Autodock-GPU


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
