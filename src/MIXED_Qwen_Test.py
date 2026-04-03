import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gc
import random
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- 🔐 LOGGED IN AUTOMATICALLY WITH YOUR TOKEN ---
login(token="YOUR_HF_TOKEN_HERE")

# =================================================================
# INTERLEAVED COGNITIVE STRESS TEST (Qwen-32B, 2000 Nodes)
# =================================================================
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
SHORT_NAME = "QWEN_MIXED_32B"

LAYER_IDX = 20         # Deep Semantic Center
N_CONCEPTS = 2000      # Ultra-Resolution Mapping
SAMPLES_PER_DOMAIN = 500 # 2,000 total mixed inferences

print(f"🧬 INITIALIZING INTERLEAVED MIXED STUDY: {MODEL_ID}")

# 1. LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

# 2. SVD KNOWLEDGE EXTRACTION
print(f"\n[1/5] Extracting Massive Knowledge Skeleton (2000 Nodes)...")
with torch.no_grad():
    W = model.model.layers[LAYER_IDX].mlp.up_proj.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    concepts = Vh[:N_CONCEPTS].cpu() 
    singular_values = S[:N_CONCEPTS].cpu().numpy()
del W, U, S
gc.collect()

# 3. BUILD THE "MIXED" DATASET (The Innovation)
print(f"\n[2/5] Building The Shuffled Cognitive Stress Test...")
benchmarks = {
    "ETHICS": ("hendrycks/ethics", "commonsense", "test"),
    "MMLU": ("cais/mmlu", "all", "test"),
    "GPQA_PhD": ("davidrein/gpqa", "gpqa_main", "train"),      
    "RedTeaming": ("Anthropic/hh-rlhf", None, "test")          
}

mixed_dataset = []

for name, (path, conf, split) in benchmarks.items():
    print(f"   -> Downloading: {name}")
    try:
        data = load_dataset(path, conf, split=split, trust_remote_code=True)
        limit = min(len(data), SAMPLES_PER_DOMAIN)
        for row in data.select(range(limit)):
            text = str(row.get('question') or row.get('text') or row.get('ctx') or row.get('Question') or row.get('chosen') or "")
            if text.strip():
                mixed_dataset.append({"domain": name, "text": text})
    except Exception as e:
        print(f"      ! Failed to load {name}: {e}")

# The most important step: Shuffling them entirely out of order
random.shuffle(mixed_dataset)
print(f"   -> Master Payload Built: {len(mixed_dataset)} fully interleaved questions.")

# 4. LIVE INTERLEAVED PROBING WITH VRAM PROTECTION
print(f"\n[3/5] Initiating Randomized Neural Assault...")
activations = {n: np.zeros(N_CONCEPTS) for n in benchmarks}

CURRENT_DOMAIN = None # Tracks the active randomized domain

def get_interleaved_hook():
    def hook(m, i, o):
        tensor = o[0] if isinstance(o, tuple) else o
        h = tensor.detach().cpu().float().mean(dim=1).squeeze()
        if len(h.shape) > 1: h = h.mean(dim=0)
        scores = torch.matmul(concepts, h).abs().numpy()
        activations[CURRENT_DOMAIN] += scores # Adds securely to the proper bucket!
    return hook

handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(get_interleaved_hook())

# Run the mixed test
for idx, item in enumerate(tqdm(mixed_dataset, desc="Scanning Brain States")):
    CURRENT_DOMAIN = item["domain"]
    
    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad(): 
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    # ⚡ VRAM CRASH PROTECTION: Aggressively flush memory every 500 operations
    if (idx + 1) % 500 == 0:
        torch.cuda.empty_cache()
        gc.collect()

handle.remove()

# 5. STRUCTURAL NORMALIZATION & LABELING
print("\n[4/5] Normalizing 2000 Interleaved Concept Data Points...")
voting = {}
for b in benchmarks:
    m, s = np.mean(activations[b]), np.std(activations[b])
    voting[b] = (activations[b] - m) / (s if s > 0 else 1)

lm_head = model.lm_head.weight.detach().float().cpu()
node_data = []

colors_map = {
    "ETHICS": "#e74c3c",       # Red
    "MMLU": "#3498db",         # Blue
    "GPQA_PhD": "#f1c40f",     # Gold 
    "RedTeaming": "#34495e"    # Dark Slate 
}

for i in tqdm(range(N_CONCEPTS), desc="Mapping Vocabulary DNA"):
    raw_logits = torch.matmul(lm_head, concepts[i])
    top_tids = torch.topk(raw_logits, 12).indices
    words = [tokenizer.decode([t]).strip() for t in top_tids]
    clean = [w for w in words if len(w) > 2 and w.isalpha()]
    lbl = " + ".join(clean[:2]) if clean else words[0]
    
    owner = max(benchmarks.keys(), key=lambda b: voting[b][i])
    node_data.append({
        "node_id": i, "label": lbl, "domain": owner, "color": colors_map[owner],
        "weight_importance": float(singular_values[i]),
        "Z_ETHICS": float(voting["ETHICS"][i]), "Z_MMLU": float(voting["MMLU"][i]),
        "Z_GPQA_PhD": float(voting["GPQA_PhD"][i]), "Z_RedTeaming": float(voting["RedTeaming"][i])
    })

csv_file = f"RESEARCH_MIXED_{SHORT_NAME}_DATA.csv"
pd.DataFrame(node_data).to_csv(csv_file, index=False)

# 6. GALAXY-SCALE METRICS VISUALIZATION
print("\n[5/5] Generating Interleaved Cognitive Map (2000 Nodes)...")
plt.figure(figsize=(40, 36))
G = nx.Graph()
for d in node_data: G.add_node(d['node_id'])

v_norm = concepts / concepts.norm(dim=1, keepdim=True)
sim = torch.mm(v_norm, v_norm.t()).abs().numpy()

# Sparsity threshold perfectly tuned for massive 2000 node layouts
thresh = np.percentile(sim, 99.75)
for i in range(N_CONCEPTS):
    for j in range(i+1, N_CONCEPTS):
        if sim[i, j] > thresh: G.add_edge(i, j)

pos = nx.spring_layout(G, k=0.08)

nx.draw_networkx_nodes(G, pos, node_size=100, node_color=[d['color'] for d in node_data], alpha=0.9, edgecolors='black', linewidths=0.1)
nx.draw_networkx_edges(G, pos, alpha=0.02, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels={d['node_id']: d['label'] for d in node_data}, font_size=3.5)

re = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=30) for k, v in colors_map.items()]
plt.legend(handles=re, loc='best', fontsize=32, framealpha=0.9, title="Mixed Domain Accuracy", title_fontsize=40)

plt.title(f"INTERLEAVED COGNITIVE ASSAULT (2000 Nodes): {MODEL_ID}", fontsize=50)
plt.axis('off')
map_file = f"MIXED_MAP_{SHORT_NAME}.png"
plt.savefig(map_file, dpi=800, bbox_inches='tight') 

print(f"\n✅ REVOLUTIONARY MIXED STUDY COMPLETE!")
print(f"Deep CSV Matrix: {csv_file}")
print(f"Ultra-HD Concept Atlas: {map_file}")
