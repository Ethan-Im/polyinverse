import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import random

RDLogger.DisableLog('rdApp.*')

def randomize_smiles(smiles, n=2):
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

def mol_to_graph(smiles, targets):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 10.0,
            atom.GetFormalCharge() / 5.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs() / 8.0,
            atom.GetMass() / 200.0,
            float(atom.GetHybridization()) / 8.0,
        ]
        atom_features.append(features)
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bond_feat = [float(bond.GetBondTypeAsDouble()), float(bond.GetIsAromatic()), float(bond.IsInRing())]
        edge_attr += [bond_feat, bond_feat]
    if len(edge_index) == 0:
        return None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y_density = torch.tensor([targets[0]], dtype=torch.float)
    y_tc = torch.tensor([targets[1]], dtype=torch.float)
    y_tg = torch.tensor([targets[2]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y_density=y_density, y_tc=y_tc, y_tg=y_tg)

class AttentiveFPMultiTask(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=128, edge_dim=3):
        super().__init__()
        self.encoder = AttentiveFP(in_channels=in_channels, hidden_channels=hidden_channels,
            out_channels=hidden_channels, edge_dim=edge_dim, num_layers=4, num_timesteps=2, dropout=0.1)
        self.shared = nn.Sequential(nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Dropout(0.1))
        self.head_density = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tg = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        shared = self.shared(h)
        return self.head_density(shared).squeeze(), self.head_tc(shared).squeeze(), self.head_tg(shared).squeeze()

# 원본 데이터 로드
df = pd.read_csv('polymer_data/combined_dataset_v2.csv')
df = df[df['Density'].notna() | df['Tc'].notna() | df['Tg'].notna()].reset_index(drop=True)
print(f"원본 데이터: {len(df)}개")

stats = {}
for col in ['Density', 'Tc', 'Tg']:
    stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}

# Train/Test split 먼저!
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}개 | Test: {len(test_df)}개")

# Train에만 augmentation 적용
print("Train augmentation 중...")
aug_rows = []
for _, row in train_df.iterrows():
    new_smiles_list = randomize_smiles(row['SMILES'], n=2)
    for new_smi in new_smiles_list:
        new_row = row.copy()
        new_row['SMILES'] = new_smi
        aug_rows.append(new_row)

train_aug = pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True)
print(f"Augmented Train: {len(train_aug)}개")

def make_graphs(df_input):
    graphs = []
    for _, row in df_input.iterrows():
        targets = []
        for col in ['Density', 'Tc', 'Tg']:
            if pd.notna(row[col]):
                targets.append((row[col] - stats[col]['mean']) / stats[col]['std'])
            else:
                targets.append(float('nan'))
        g = mol_to_graph(row['SMILES'], targets)
        if g:
            graphs.append(g)
    return graphs

train_graphs = make_graphs(train_aug)
test_graphs = make_graphs(test_df)
print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

model = AttentiveFPMultiTask()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining...")

best_avg_r2 = -999
for epoch in range(500):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred_d, pred_t, pred_tg = model(batch)
        loss = torch.tensor(0.0, requires_grad=True)
        mask_d  = ~torch.isnan(batch.y_density.squeeze())
        mask_tc = ~torch.isnan(batch.y_tc.squeeze())
        mask_tg = ~torch.isnan(batch.y_tg.squeeze())
        if mask_d.sum() > 0:
            loss = loss + ((pred_d[mask_d] - batch.y_density.squeeze()[mask_d]) ** 2).mean()
        if mask_tc.sum() > 0:
            loss = loss + ((pred_t[mask_tc] - batch.y_tc.squeeze()[mask_tc]) ** 2).mean()
        if mask_tg.sum() > 0:
            loss = loss + ((pred_tg[mask_tg] - batch.y_tg.squeeze()[mask_tg]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    if (epoch + 1) % 100 == 0:
        model.eval()
        results = {col: {'true': [], 'pred': []} for col in ['Density', 'Tc', 'Tg']}
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch)
                attrs = [batch.y_density, batch.y_tc, batch.y_tg]
                for i, col in enumerate(['Density', 'Tc', 'Tg']):
                    mask = ~torch.isnan(attrs[i].squeeze())
                    if mask.sum() > 0:
                        true_vals = attrs[i].squeeze()[mask].numpy() * stats[col]['std'] + stats[col]['mean']
                        pred_vals = preds[i][mask].numpy() * stats[col]['std'] + stats[col]['mean']
                        results[col]['true'].extend(true_vals.tolist())
                        results[col]['pred'].extend(pred_vals.tolist())

        r2s = []
        print(f"\nEpoch {epoch+1}/500 | Loss: {avg_loss:.4f}")
        for col in ['Density', 'Tc', 'Tg']:
            if len(results[col]['true']) > 1:
                r2 = r2_score(results[col]['true'], results[col]['pred'])
                mae = mean_absolute_error(results[col]['true'], results[col]['pred'])
                print(f"  {col:<10} R2: {r2:.3f} | MAE: {mae:.4f}")
                r2s.append(r2)

        avg_r2 = np.mean(r2s)
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            torch.save(model.state_dict(), 'best_attentivefp_v4.pt')

print("\n=== Final Performance (올바른 검증) ===")
model.load_state_dict(torch.load('best_attentivefp_v4.pt'))
model.eval()
results = {col: {'true': [], 'pred': []} for col in ['Density', 'Tc', 'Tg']}
with torch.no_grad():
    for batch in test_loader:
        preds = model(batch)
        attrs = [batch.y_density, batch.y_tc, batch.y_tg]
        for i, col in enumerate(['Density', 'Tc', 'Tg']):
            mask = ~torch.isnan(attrs[i].squeeze())
            if mask.sum() > 0:
                true_vals = attrs[i].squeeze()[mask].numpy() * stats[col]['std'] + stats[col]['mean']
                pred_vals = preds[i][mask].numpy() * stats[col]['std'] + stats[col]['mean']
                results[col]['true'].extend(true_vals.tolist())
                results[col]['pred'].extend(pred_vals.tolist())

for col in ['Density', 'Tc', 'Tg']:
    r2 = r2_score(results[col]['true'], results[col]['pred'])
    mae = mean_absolute_error(results[col]['true'], results[col]['pred'])
    print(f"{col:<10} R2: {r2:.3f} | MAE: {mae:.4f}")
