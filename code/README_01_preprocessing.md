# 01. Checking Batch Difference

### Dataset overview

- Five batches: smiles_batch_1.csv … smiles_batch_5.csv.
- Each batch = 20,032 molecules (Except Batch_2 = 20,029).
- Main molecular features analyzed: MW, logP, TPSA, RotB.

### Summary statistics

- MW: means ≈ 334–339 Da, SD ≈ 39–41.
- logP: means ≈ 2.37–2.62, but extremely large max values (~500).
- TPSA: max values ≈ 146–159, small but significant batch differences.
- RotB: mean ≈ 4.2, SD ≈ 1.44, max 11.

<img width="1005" height="505" alt="image" src="https://github.com/user-attachments/assets/e3c003ad-d3c0-4ff9-9010-237b8b3fde73" />

### Statistical tests

- Kruskal–Wallis shows significant batch differences for MW, logP, and TPSA (p < 0.01).
- Post-hoc (Bonferroni) found multiple pairwise batch differences.

