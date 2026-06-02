import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from rdkit.Chem import RDConfig
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.explain import Explainer, GNNExplainer
from huggingface_hub import hf_hub_download
import gradio as gr
import os, sys, random, io, re
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

RDLogger.DisableLog('rdApp.*')

def get_sa_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        return sascorer.calculateScore(mol)
    except:
        return 10.0

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
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
        return None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch), mol

class AttentiveFPMultiTask(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=128, edge_dim=3):
        super().__init__()
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
            nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Dropout(0.1))
        self.head_density = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tg = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        h = self.encoder(x, edge_index, edge_attr, batch)
        shared = self.shared(h)
        return (self.head_density(shared).squeeze(),
                self.head_tc(shared).squeeze(),
                self.head_tg(shared).squeeze())

class PolymerVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, latent_dim=128, max_len=100):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embed_dim + latent_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def sample(self, n=10, device='cpu', token2idx=None, idx2token=None):
        z = torch.randn(n, self.latent_dim).to(device)
        sos_idx = token2idx['<SOS>']
        h = torch.tanh(self.fc_decode(z)).unsqueeze(0).repeat(2, 1, 1)
        inp_token = torch.tensor([sos_idx] * n, dtype=torch.long).to(device)
        generated = []
        for _ in range(self.max_len):
            emb = self.embed(inp_token).unsqueeze(1)
            z_exp = z.unsqueeze(1)
            inp = torch.cat([emb, z_exp], dim=-1)
            out, h = self.decoder_rnn(inp, h)
            logits = self.fc_out(out.squeeze(1))
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            inp_token = next_token
        generated = torch.stack(generated, dim=1)
        results = []
        for seq in generated:
            tokens = []
            for idx in seq:
                token = idx2token.get(idx.item(), '')
                if token == '<EOS>':
                    break
                if token not in ['<PAD>', '<SOS>']:
                    tokens.append(token)
            smi = ''.join(tokens)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                results.append(Chem.MolToSmiles(mol))
        return results

# Load data stats
df = pd.read_csv('polymer_data/combined_dataset_v2.csv')
stats = {}
for col in ['Density', 'Tc', 'Tg']:
    stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}

# Load AttentiveFP model
model = AttentiveFPMultiTask(in_channels=8, hidden_channels=128, edge_dim=3)
model_path = hf_hub_download(
    repo_id='Ethan-Im/polyinverse-model',
    filename='best_attentivefp_combined.pt',
    repo_type='model'
)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Load VAE model
vae_vocab_path = hf_hub_download(repo_id='Ethan-Im/polyinverse-model', filename='vae_vocab.pt', repo_type='model')
vae_vocab = torch.load(vae_vocab_path)
vae_token2idx = vae_vocab['token2idx']
vae_idx2token = vae_vocab['idx2token']
vae_vocab_size = vae_vocab['vocab_size']

vae_model = PolymerVAE(vocab_size=vae_vocab_size)
vae_model_path = hf_hub_download(repo_id='Ethan-Im/polyinverse-model', filename='best_vae.pt', repo_type='model')
vae_model.load_state_dict(torch.load(vae_model_path, map_location='cpu'))
vae_model.eval()

# Explainer
class DensityWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x, edge_index, edge_attr, batch=None):
        pred_d, _, _ = self.base(x, edge_index, edge_attr, batch)
        return pred_d

density_model = DensityWrapper(model)

explainer = Explainer(
    model=density_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw'),
)

def predict(smiles):
    graph, mol = mol_to_graph(smiles)
    if graph is None or mol is None:
        return "Invalid SMILES", None
    with torch.no_grad():
        pred_d, pred_tc, pred_tg = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        density = pred_d.item() * stats['Density']['std'] + stats['Density']['mean']
        tc = pred_tc.item() * stats['Tc']['std'] + stats['Tc']['mean']
        tg = pred_tg.item() * stats['Tg']['std'] + stats['Tg']['mean']
    img = Draw.MolToImage(mol, size=(300, 300))
    result = f"""## Prediction Results
**SMILES:** `{smiles}`

| Property | Predicted |
|----------|-----------|
| Density  | {density:.4f} g/ml |
| Tc       | {tc:.4f} (normalized) |
| Tg       | {tg:.2f} C |

*Model: AttentiveFP Multi-task (R2 Density=0.871, Tc=0.728, Tg=0.793)*
"""
    return result, img

def explain(smiles):
    graph, mol = mol_to_graph(smiles)
    if graph is None or mol is None:
        return "Invalid SMILES", None
    explanation = explainer(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
    )
    node_importance = explanation.node_mask.sum(dim=1).detach().numpy()
    node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
    cmap = plt.cm.RdYlGn
    atom_colors = {}
    atom_radii = {}
    for i, score in enumerate(node_importance):
        rgba = cmap(float(score))
        atom_colors[i] = (rgba[0], rgba[1], rgba[2])
        atom_radii[i] = 0.3 + 0.5 * float(score)
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    feature_names = ['AtomicNum', 'Degree', 'Charge', 'Aromatic', 'InRing', 'NumHs', 'Mass', 'Hybridization']
    top_atoms = sorted(enumerate(node_importance), key=lambda x: x[1], reverse=True)[:3]
    text = "### Why this prediction?\n\n"
    text += "Green = High importance | Red = Low importance\n\n"
    text += "**Top influential atoms:**\n"
    for idx, score in top_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        aromatic = "aromatic " if atom.GetIsAromatic() else ""
        ring = "in ring " if atom.IsInRing() else ""
        text += f"- **Atom {idx} ({symbol})**: importance {score:.3f} ({aromatic}{ring})\n"
    feat_importance = explanation.node_mask.abs().mean(dim=0).detach().numpy()
    feat_importance = feat_importance / feat_importance.sum()
    top_features = sorted(zip(feature_names, feat_importance), key=lambda x: x[1], reverse=True)[:4]
    text += "\n**Key molecular features driving prediction:**\n"
    for feat, imp in top_features:
        bar = "X" * int(imp * 20)
        text += f"- {feat}: {bar} ({imp:.3f})\n"
    return text, img

def mutate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rw_mol = Chem.RWMol(mol)
        atom_idx = random.randint(0, mol.GetNumAtoms()-1)
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        candidates = [6, 7, 8]
        candidates = [c for c in candidates if c != atom.GetAtomicNum()]
        atom.SetAtomicNum(random.choice(candidates))
        new_smiles = Chem.MolToSmiles(rw_mol)
        if Chem.MolFromSmiles(new_smiles):
            return new_smiles
    except:
        pass
    return None

def predict_properties(smiles):
    graph, mol = mol_to_graph(smiles)
    if graph is None:
        return None, None, None
    with torch.no_grad():
        pred_d, pred_tc, pred_tg = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        density = pred_d.item() * stats['Density']['std'] + stats['Density']['mean']
        tc = pred_tc.item() * stats['Tc']['std'] + stats['Tc']['mean']
        tg = pred_tg.item() * stats['Tg']['std'] + stats['Tg']['mean']
    return density, tc, tg

def inverse_design(target_density, target_tc, target_tg, w_density, w_tc, w_tg, tolerance, n_iterations):
    population = df['SMILES'].dropna().sample(min(50, len(df)), random_state=42).tolist()
    best_candidates = []
    for _ in range(int(n_iterations)):
        scores = []
        for smiles in population:
            density, tc, tg = predict_properties(smiles)
            if density is None:
                continue
            sa = get_sa_score(smiles)
            err_d = abs(density - target_density) * w_density
            err_tc = abs(tc - target_tc) * w_tc
            err_tg = abs(tg - target_tg) * w_tg
            combined = err_d + err_tc + err_tg + max(0, sa - 4.0) * 0.1
            scores.append((smiles, density, tc, tg, combined, sa))
        scores.sort(key=lambda x: x[4])
        for smiles, d, tc, tg, combined, sa in scores:
            err = abs(d - target_density)
            if err <= tolerance and sa <= 4.0:
                if smiles not in [c[0] for c in best_candidates]:
                    best_candidates.append((smiles, d, tc, tg, err, sa))
        survivors = [s[0] for s in scores[:25]]
        new_population = survivors.copy()
        while len(new_population) < 50:
            parent = random.choice(survivors)
            mutated = mutate_smiles(parent)
            new_population.append(mutated if mutated else random.choice(survivors))
        population = new_population
    if not best_candidates:
        return "No candidates found. Try increasing tolerance.", None
    best_candidates.sort(key=lambda x: x[4])
    top = best_candidates[:5]
    result = f"## Inverse Design Results\n**Targets:** Density={target_density:.3f} | Tc={target_tc:.1f} | Tg={target_tg:.1f}C\n\n"
    result += "| SMILES | Density | Tc | Tg | Error | SA |\n|--------|---------|----|----|-------|----|\n"
    for smiles, d, tc, tg, err, sa in top:
        result += f"| `{smiles[:30]}` | {d:.4f} | {tc:.1f} | {tg:.1f} | {err:.4f} | {sa:.2f} |\n"
    best_mol = Chem.MolFromSmiles(top[0][0])
    img = Draw.MolToImage(best_mol, size=(300, 300)) if best_mol else None
    return result, img

def generate_molecules(n_samples, sa_threshold):
    with torch.no_grad():
        candidates = vae_model.sample(
            n=50, device='cpu',
            token2idx=vae_token2idx,
            idx2token=vae_idx2token
        )
    results = []
    for smi in candidates:
        sa = get_sa_score(smi)
        if sa <= sa_threshold:
            density, tc, tg = predict_properties(smi)
            if density is not None:
                results.append((smi, density, tc, tg, sa))
        if len(results) >= int(n_samples):
            break
    if not results:
        return "No valid molecules found. Try increasing SA threshold.", None
    result = "## Generated Molecules (VAE)\n\n"
    result += "| SMILES | Density | Tc | Tg | SA Score |\n|--------|---------|----|----|----------|\n"
    for smi, d, tc, tg, sa in results:
        result += f"| `{smi[:35]}` | {d:.4f} | {tc:.3f} | {tg:.1f} | {sa:.2f} |\n"
    best_mol = Chem.MolFromSmiles(results[0][0])
    img = Draw.MolToImage(best_mol, size=(300, 300)) if best_mol else None
    return result, img

examples = ["*CC(c1ccccc1)*", "*CC(C(=O)OC)*", "*CC(Cl)*", "*C(F)(F)*", "*CC*"]

with gr.Blocks(title="Polyinverse v5") as demo:
    gr.Markdown("""
# Polyinverse v5
## AI-powered Polymer Property Predictor & Inverse Design
*AttentiveFP Multi-task | R2 Density=0.871, Tc=0.728, Tg=0.793*
    """)
    with gr.Tabs():
        with gr.Tab("Forward Prediction"):
            gr.Markdown("### Predict Density, Tc, Tg from SMILES")
            with gr.Row():
                with gr.Column():
                    smiles_input = gr.Textbox(label="Polymer SMILES", value="*CC(c1ccccc1)*")
                    predict_btn = gr.Button("Predict", variant="primary")
                    gr.Markdown("**Examples:**")
                    for smi in examples:
                        gr.Button(smi).click(fn=lambda s=smi: s, outputs=smiles_input)
                with gr.Column():
                    output_text = gr.Markdown()
                    output_img = gr.Image(label="Molecule")
            predict_btn.click(predict, inputs=smiles_input, outputs=[output_text, output_img])
            smiles_input.submit(predict, inputs=smiles_input, outputs=[output_text, output_img])

        with gr.Tab("Why This Polymer?"):
            gr.Markdown("### Explainability — Which atoms drive the prediction?")
            with gr.Row():
                with gr.Column():
                    explain_input = gr.Textbox(label="Polymer SMILES", value="*CC(c1ccccc1)*")
                    explain_btn = gr.Button("Explain", variant="primary")
                    gr.Markdown("*Note: Takes ~10 seconds to compute*")
                with gr.Column():
                    explain_text = gr.Markdown()
                    explain_img = gr.Image(label="Atom Importance Map")
            explain_btn.click(explain, inputs=explain_input, outputs=[explain_text, explain_img])

        with gr.Tab("Inverse Design"):
            gr.Markdown("### Design molecules with target properties")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Target Properties**")
                    target_density = gr.Slider(0.8, 2.0, value=1.2, step=0.05, label="Target Density (g/ml)")
                    target_tc = gr.Slider(0, 500, value=200, step=10, label="Target Tc")
                    target_tg = gr.Slider(-100, 300, value=100, step=10, label="Target Tg (C)")
                    gr.Markdown("**Property Weights**")
                    w_density = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Density Weight")
                    w_tc = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Tc Weight")
                    w_tg = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Tg Weight")
                    gr.Markdown("**Search Settings**")
                    tolerance = gr.Slider(0.01, 0.2, value=0.08, step=0.01, label="Tolerance")
                    n_iter = gr.Slider(100, 500, value=200, step=100, label="Iterations")
                    design_btn = gr.Button("Design", variant="primary")
                with gr.Column():
                    design_output = gr.Markdown()
                    design_img = gr.Image(label="Best Candidate")
            design_btn.click(
                inverse_design,
                inputs=[target_density, target_tc, target_tg, w_density, w_tc, w_tg, tolerance, n_iter],
                outputs=[design_output, design_img]
            )

        with gr.Tab("VAE Molecule Generation"):
            gr.Markdown("### Generate novel polymer candidates using Variational Autoencoder")
            with gr.Row():
                with gr.Column():
                    n_samples = gr.Slider(1, 10, value=5, step=1, label="Number of candidates")
                    sa_thresh = gr.Slider(2.0, 6.0, value=4.0, step=0.5, label="SA Score threshold")
                    generate_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    gen_output = gr.Markdown()
                    gen_img = gr.Image(label="Best Generated Molecule")
            generate_btn.click(
                generate_molecules,
                inputs=[n_samples, sa_thresh],
                outputs=[gen_output, gen_img]
            )

demo.launch(share=True)
