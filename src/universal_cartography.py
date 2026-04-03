import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# =================================================================
# UNIVERSAL ENGINE (V2.0 - CLEAN & QUIET EDITION)
# =================================================================
# TO SWITCH MODELS: 
# Paste one of: "upstage/SOLAR-10.7B-v1.0" OR "NousResearch/Hermes-3-Llama-3.1-8B"
MODEL_ID = "upstage/SOLAR-10.7B-v1.0" 

# AUTO-NAMING
SHORT_NAME = MODEL_ID.split('/')[-1].replace("-", "_")
N_CONCEPTS = 250
MAX_SAMPLES = 200 # Higher samples = deeper research

print(f"🌍 STARTING UNIFIED STUDY: {MODEL_ID}")

# 1. LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

# 2. AUTO-LAYER DETECTION
total_layers = len(model.model.layers)
LAYER_IDX = total_layers // 2 + 3 
print(f"   -> Probing Layer {LAYER_IDX} of {total_layers}")

# 3. SVD WEIGHT EXTRACTION
with torch.no_grad():
    # Detects Llama/Solar/Qwen MLP structures automatically
    W = model.model.layers[LAYER_IDX].mlp.up_proj.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    concepts = Vh[:N_CONCEPTS].cpu() 
del W, U, S
gc.collect()

# 4. ROBUST PROBING
benchmarks = {
    "TruthfulQA": ("truthful_qa", "generation", "validation"),
    "ETHICS": ("hendrycks/ethics", "commonsense", "test"),
    "MMLU": ("cais/mmlu", "all", "test"),
    "Logic": ("hellaswag", None, "train")
}

activations = {n: np.zeros(N_CONCEPTS) for n in benchmarks}

def get_hook(name):
    def hook(m, i, o):
        tensor = o[0] if isinstance(o, tuple) else o
        h = tensor.detach().cpu().float().mean(dim=1).squeeze()
        if len(h.shape) > 1: h = h.mean(dim=0)
        scores = torch.matmul(concepts, h)
        activations[name] += scores.abs().numpy()
    return hook

for name, (path, conf, split) in benchmarks.items():
    print(f"\n   -> BENCHMARKING: {name}...")
    try:
        data = load_dataset(path, conf, split=split, trust_remote_code=True).select(range(MAX_SAMPLES))
        handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(get_hook(name))
        
        # tqdm for a nice clean visual progress bar
        for row in tqdm(data, leave=False):
            t = str(row.get('question') or row.get('text') or row.get('ctx') or "")
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad(): 
                # SILENCED WARNINGS HERE (pad_token_id set)
                model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
    except Exception as e: print(f"      ! Warning: {name} skip: {e}")

# 5. BALANCING
print("\n[Analysis] Normalizing High-Resolution Data...")
voting = {}
for b in benchmarks:
    m, s = np.mean(activations[b]), np.std(activations[b])
    voting[b] = (activations[b] - m) / (s if s > 0 else 1)

lm_head = model.lm_head.weight.detach().float().cpu()
node_data = []
colors_map = {"TruthfulQA": "#2ecc71", "ETHICS": "#e74c3c", "MMLU": "#3498db", "Logic": "#9b59b6"}

# 6. QUALITY LABELING
print("\n[Study] Extracting Concept Semantics...")
for i in range(N_CONCEPTS):
    raw_logits = torch.matmul(lm_head, concepts[i])
    top_tids = torch.topk(raw_logits, 10).indices
    words = [tokenizer.decode([t]).strip() for t in top_tids if len(tokenizer.decode([t]).strip()) > 1]
    lbl = " + ".join(words[:2]) if words else "Unknown"
    owner = max(benchmarks.keys(), key=lambda b: voting[b][i])
    node_data.append({"id": i, "label": lbl, "color": colors_map[owner]})

# 7. FINAL EXPORT (Auto-Naming)
data_file = f"RESEARCH_{SHORT_NAME}_DATA.csv"
map_file = f"MAP_{SHORT_NAME}.png"
pd.DataFrame(node_data).to_csv(data_file, index=False)

plt.figure(figsize=(24, 20))
G = nx.Graph()
for d in node_data: G.add_node(d['id'])
v_norm = concepts / concepts.norm(dim=1, keepdim=True)
sim = torch.mm(v_norm, v_norm.t()).abs().numpy()
thresh = np.percentile(sim, 96.5)
for i in range(N_CONCEPTS):
    for j in range(i+1, N_CONCEPTS):
        if sim[i, j] > thresh: G.add_edge(i, j)

pos = nx.spring_layout(G, k=0.18)
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=[d['color'] for d in node_data], alpha=0.9, edgecolors='black', linewidths=0.2)
nx.draw_networkx_labels(G, pos, labels={d['id']: d['label'] for d in node_data}, font_size=5)
le = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=15) for k, v in colors_map.items()]
plt.legend(handles=le, loc='best', fontsize=16)

plt.title(f"Functional Neural Atlas: {MODEL_ID}", fontsize=30)
plt.axis('off')
plt.savefig(map_file, dpi=400)

print(f"\n✅ COMPLETED FOR {MODEL_ID}!")
print(f"Files Generated: '{data_file}' and '{map_file}'")
