import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# 원본 데이터로 train/test split
df_orig = pd.read_csv('polymer_data/combined_dataset_v2.csv')
df_orig = df_orig[df_orig['Density'].notna() | df_orig['Tc'].notna() | df_orig['Tg'].notna()].reset_index(drop=True)

# 원본 canonical SMILES 기준으로 split
from sklearn.model_selection import train_test_split
train_orig, test_orig = train_test_split(df_orig, test_size=0.2, random_state=42)

# augmented 데이터에서 test 원본과 겹치는 거 확인
df_aug = pd.read_csv('polymer_data/combined_dataset_v3.csv')

# canonical SMILES로 변환
def canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol)
    return smi

test_canonical = set(test_orig['SMILES'].apply(canonical))
aug_canonical = set(df_aug['SMILES'].apply(canonical))

overlap = test_canonical & aug_canonical
print(f"Test 원본: {len(test_canonical)}개")
print(f"Augmented 전체: {len(aug_canonical)}개")
print(f"겹치는 분자: {len(overlap)}개")
print(f"겹치는 비율: {len(overlap)/len(test_canonical)*100:.1f}%")
