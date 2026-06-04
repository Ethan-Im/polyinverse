import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import random

RDLogger.DisableLog('rdApp.*')

def randomize_smiles(smiles, n=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    results = set()
    for _ in range(n * 5):
        atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(atom_order)
        try:
            new_mol = Chem.RenumberAtoms(mol, atom_order)
            new_smi = Chem.MolToSmiles(new_mol, canonical=False)
            if new_smi and Chem.MolFromSmiles(new_smi):
                results.add(new_smi)
        except:
            pass
        if len(results) >= n:
            break
    return list(results)

df = pd.read_csv('polymer_data/combined_dataset_v2.csv')
print(f"원본 데이터: {len(df)}개")

augmented = []
for i, row in df.iterrows():
    if i % 1000 == 0:
        print(f"  {i}/{len(df)}")
    new_smiles_list = randomize_smiles(row['SMILES'], n=2)
    for new_smi in new_smiles_list:
        new_row = row.copy()
        new_row['SMILES'] = new_smi
        augmented.append(new_row)

df_aug = pd.DataFrame(augmented)
df_combined = pd.concat([df, df_aug], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['SMILES']).reset_index(drop=True)

print(f"증강 후 데이터: {len(df_combined)}개")
print(df_combined[['Density','Tc','Tg']].notna().sum())

df_combined.to_csv('polymer_data/combined_dataset_v3.csv', index=False)
print("저장 완료: combined_dataset_v3.csv")
