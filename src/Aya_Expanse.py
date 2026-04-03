import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gc
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- 🔐 LOGGED IN AUTOMATICALLY WITH YOUR TOKEN ---
login(token="YOUR_HF_TOKEN_HERE")

# =================================================================
# ULTRA-RESOLUTION NEUROCARTOGRAPHY (2,000 NODES / 6 BENCHMARKS)
# =================================================================
MODEL_ID = "CohereForAI/aya-expanse-32b"
SHORT_NAME = "AYA_EXPANSE_32B"

LAYER_IDX = 22         # Deep Semantic Layer
N_CONCEPTS = 2000      # ⚡ ULTRA-RESOLUTION MAPPING
MAX_SAMPLES = 2000     # 12,000 total benchmark inferences

print(f"🌌 INITIALIZING ULTRA-RESOLUTION STUDY: {MODEL_ID}")

# 1. LOAD EXPANSIVE MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

# 2. MASSIVE WEIGHT EXTRACTION (SVD)
print(f"\n[1/5] Extracting Massive Knowledge Skeleton (2,000 Nodes)...")
with torch.no_grad():
    W = model.model.layers[LAYER_IDX].mlp.up_proj.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    concepts = Vh[:N_CONCEPTS].cpu() 
    singular_values = S[:N_CONCEPTS].cpu().numpy()
del W, U, S
gc.collect()

# 3. ROBUST 6-DOMAIN PROBING
print(f"\n[2/5] Live Probing 6 Benchmarks ({MAX_SAMPLES} samples each)...")
benchmarks = {
    "TruthfulQA": ("truthful_qa", "generation", "validation"),
    "ETHICS": ("hendrycks/ethics", "commonsense", "test"),
    "MMLU": ("cais/mmlu", "all", "test"),
    "Logic": ("hellaswag", None, "train"),
    "GPQA_PhD": ("davidrein/gpqa", "gpqa_main", "train"),      
    "RedTeaming": ("Anthropic/hh-rlhf", None, "test")          
}

activations = {n: np.zeros(N_CONCEPTS) for n in benchmarks}

def get_ultra_hook(name):
    def hook(m, i, o):
        tensor = o[0] if isinstance(o, tuple) else o
        h = tensor.detach().cpu().float().mean(dim=1).squeeze()
        if len(h.shape) > 1: h = h.mean(dim=0)
        scores = torch.matmul(concepts, h)
        activations[name] += scores.abs().numpy()
    return hook

for name, (path, conf, split) in benchmarks.items():
    print(f"\n   -> BENCHMARK: {name} (Scanning 2,000 Brain Sectors)...")
    try:
        data = load_dataset(path, conf, split=split, trust_remote_code=True)
        limit = min(len(data), MAX_SAMPLES)
        subset_data = data.select(range(limit))
        
        handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(get_ultra_hook(name))
        
        for row in tqdm(subset_data, leave=False):
            t = str(row.get('question') or row.get('text') or row.get('ctx') or row.get('Question') or row.get('chosen') or "")
            if not t.strip(): continue
            
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            with torch.no_grad(): 
                model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        handle.remove()
    except Exception as e: 
        print(f"      ! Warning: Problem processing dataset {name}: {e}")

# 4. ULTRA-DEEP NORMALIZATION
print("\n[3/5] Normalizing 12,000 Cognitive Data Points...")
voting = {}
for b in benchmarks:
    m, s = np.mean(activations[b]), np.std(activations[b])
    voting[b] = (activations[b] - m) / (s if s > 0 else 1)

lm_head = model.lm_head.weight.detach().float().cpu()
node_data = []

colors_map = {
    "TruthfulQA": "#2ecc71",   # Green
    "ETHICS": "#e74c3c",       # Red
    "MMLU": "#3498db",         # Blue
    "Logic": "#9b59b6",        # Purple
    "GPQA_PhD": "#f1c40f",     # Gold 
    "RedTeaming": "#34495e"    # Dark Slate 
}

# 5. STRUCTURAL LABELING & MASSIVE CSV
print("\n[4/5] Labeling 2000 Nodes and Generating Massive CSV...")
for i in tqdm(range(N_CONCEPTS), desc="Extracting Conceptual DNA"):
    raw_logits = torch.matmul(lm_head, concepts[i])
    top_tids = torch.topk(raw_logits, 15).indices
    
    words = [tokenizer.decode([t]).strip() for t in top_tids]
    clean_words = [w for w in words if len(w) > 2 and w.isalpha()]
    lbl = " + ".join(clean_words[:2]) if clean_words else words[0]
    
    owner = max(benchmarks.keys(), key=lambda b: voting[b][i])
    
    row_data = {
        "node_id": i,
        "structural_label": lbl,
        "dominant_domain": owner,
        "structural_importance (Singular Value)": float(singular_values[i]),
        "Z_TruthfulQA": float(voting["TruthfulQA"][i]),
        "Z_ETHICS": float(voting["ETHICS"][i]),
        "Z_MMLU": float(voting["MMLU"][i]),
        "Z_Logic": float(voting["Logic"][i]),
        "Z_GPQA_PhD": float(voting["GPQA_PhD"][i]),
        "Z_RedTeaming": float(voting["RedTeaming"][i])
    }
    node_data.append(row_data)

csv_file = f"RESEARCH_DEEP_{SHORT_NAME}_DATA.csv"
pd.DataFrame(node_data).to_csv(csv_file, index=False)

# 6. GALAXY-SCALE GRAPH VISUALIZATION
print("\n[5/5] Generating Galaxy-Scale Visual Knowledge Map (May take a minute for 2000 nodes)...")
# Massive figure size to accommodate 2000 nodes mapping
plt.figure(figsize=(46, 40))
G = nx.Graph()
for d in node_data: G.add_node(d['node_id'])

v_norm = concepts / concepts.norm(dim=1, keepdim=True)
sim = torch.mm(v_norm, v_norm.t()).abs().numpy()

# Extremely high sparsity threshold (99.7th percentile) to prevent the graph from becoming a solid ball
thresh = np.percentile(sim, 99.7)
for i in range(N_CONCEPTS):
    for j in range(i+1, N_CONCEPTS):
        if sim[i, j] > thresh: G.add_edge(i, j)

pos = nx.spring_layout(G, k=0.08)

node_colors = [colors_map[d['dominant_domain']] for d in node_data]
nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors, alpha=0.85, edgecolors='black', linewidths=0.1)
nx.draw_networkx_edges(G, pos, alpha=0.02, edge_color='gray')

# Text must be incredibly small to fit 2,000 items
nx.draw_networkx_labels(G, pos, labels={d['node_id']: d['structural_label'] for d in node_data}, font_size=3.5, font_color='black')

re = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=30) for k, v in colors_map.items()]
plt.legend(handles=re, loc='upper right', fontsize=32, framealpha=0.9, title="Cognitive Domain", title_fontsize=40)

plt.title(f"ULTRA-RESOLUTION MULTILINGUAL MAP (2000 Nodes): {MODEL_ID}", fontsize=60)
plt.axis('off')
map_file = f"ULTRA_MAP_{SHORT_NAME}.png"

# Ultra-High DPI for extreme zooming capability
plt.savefig(map_file, dpi=800, bbox_inches='tight') 

print(f"\n✅ REVOLUTIONARY STUDY COMPLETE FOR {MODEL_ID}!")
print(f"Massive CSV Matrix: {csv_file}")
print(f"Ultra-Resolution Visual Atlas: {map_file}")
