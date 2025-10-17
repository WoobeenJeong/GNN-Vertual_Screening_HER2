# Checking Splited data Differences

### Dataset overview

- Five batches: smiles_batch_1.csv … smiles_batch_5.csv.
- Each batch = 20,032 molecules (Except Batch_2 = 20,029).
- Main molecular features analyzed: MW, logP, TPSA, RotB.

### Summary statistics

- MW: means ≈ 334–339 Da, SD ≈ 39–41.
- logP: means ≈ 2.37–2.62, but extremely large max values.
- TPSA: max values ≈ 146–159, small but significant batch differences.
- RotB: mean ≈ 4.2, SD ≈ 1.44, max 11.

<img width="3189" height="852" alt="image" src="https://github.com/user-attachments/assets/294b1245-7eee-4573-ae6e-fc69f6378312" />

### Statistical tests

- Kruskal–Wallis shows significant batch differences for MW, logP, and TPSA (p < 0.01).
- Post-hoc (Bonferroni) found multiple pairwise batch differences.

<img width="2922" height="774" alt="image" src="https://github.com/user-attachments/assets/d6c76588-97b0-4f24-abf5-20b707e718ce" />
