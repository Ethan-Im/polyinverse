import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

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
        bond_feat = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsAromatic()),
            float(bond.IsInRing()),
        ]
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
        super(AttentiveFPMultiTask, self).__init__()
        self.encoder = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=4,
            num_timesteps=2,
            dropout=0.1,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head_density = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tg = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        shared = self.shared(h)
        return (self.head_density(shared).squeeze(),
                self.head_tc(shared).squeeze(),
                self.head_tg(shared).squeeze())

df = pd.read_csv('polymer_data/combined_dataset_v3.csv')
df = df[df['Density'].notna() | df['Tc'].notna() | df['Tg'].notna()].reset_index(drop=True)
print(f"Total samples: {len(df)}")

stats = {}
for col in ['Density', 'Tc', 'Tg']:
    stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}
    print(f"{col}: mean={stats[col]['mean']:.3f}, std={stats[col]['std']:.3f}")

graphs = []
for _, row in df.iterrows():
    targets = []
    for col in ['Density', 'Tc', 'Tg']:
        if pd.notna(row[col]):
            targets.append((row[col] - stats[col]['mean']) / stats[col]['std'])
        else:
            targets.append(float('nan'))
    g = mol_to_graph(row['SMILES'], targets)
    if g:
        graphs.append(g)

print(f"Valid graphs: {len(graphs)}")

train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = AttentiveFPMultiTask(in_channels=8, hidden_channels=128, edge_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining AttentiveFP Multi-task...")

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
                unit = 'g/ml' if col == 'Density' else 'C' if col == 'Tg' else 'norm'
                print(f"  {col:<10} R2: {r2:.3f} | MAE: {mae:.4f} {unit}")
                r2s.append(r2)

        avg_r2 = np.mean(r2s)
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            torch.save(model.state_dict(), 'best_attentivefp_combined.pt')

model.load_state_dict(torch.load('best_attentivefp_combined.pt'))
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

print(f"\n=== AttentiveFP Multi-task Final Performance ===")
for col in ['Density', 'Tc', 'Tg']:
    r2 = r2_score(results[col]['true'], results[col]['pred'])
    mae = mean_absolute_error(results[col]['true'], results[col]['pred'])
    unit = 'g/ml' if col == 'Density' else 'C' if col == 'Tg' else 'norm'
    print(f"{col:<10} R2: {r2:.3f} | MAE: {mae:.4f} {unit}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['steelblue', 'darkorange', 'green']
for i, (col, color) in enumerate(zip(['Density', 'Tc', 'Tg'], colors)):
    t = results[col]['true']
    p = results[col]['pred']
    r2 = r2_score(t, p)
    axes[i].scatter(t, p, alpha=0.4, color=color, s=15)
    axes[i].plot([min(t), max(t)], [min(t), max(t)], 'r--')
    axes[i].set_title(f'{col} (R2={r2:.3f})')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
plt.tight_layout()
plt.savefig('paper_figures/attentivefp_results.png', dpi=300)
print("Saved: paper_figures/attentivefp_results.png")
