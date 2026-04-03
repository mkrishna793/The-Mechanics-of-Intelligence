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
# DETAILED MASTER SCRIPT: REFLECTION-LLAMA-3.1-8B
# =================================================================
MODEL_ID = MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"
SHORT_NAME = "REFLECTION_8B"

LAYER_IDX = 20         # Deep logical layer in Llama-3 architecture
N_CONCEPTS = 2000      # 2000 Detailed Nodes
SAMPLES_PER_DOMAIN = 500 

print(f"🧠 INITIALIZING DETAILED REFLECTION MAP MAPPING: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

print(f"\n[1/5] Extracting Massive Knowledge Skeleton (2000 Nodes)...")
with torch.no_grad():
    # Llama-3.1 architecture uses standard up_proj
    W = model.model.layers[LAYER_IDX].mlp.up_proj.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    concepts = Vh[:N_CONCEPTS].cpu() 
    singular_values = S[:N_CONCEPTS].cpu().numpy()
del W, U, S
gc.collect()

print(f"\n[2/5] Building Open Datasets...")
benchmarks = {
    "Math_Logic": ("gsm8k", "main", "train"),
    "MMLU": ("cais/mmlu", "all", "test"),
    "TruthfulQA": ("truthful_qa", "generation", "validation"),      
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

random.shuffle(mixed_dataset)
print(f"   -> Sequence Built: {len(mixed_dataset)} interleaved stress-tests.")

print(f"\n[3/5] Initiating Reflection Sequence...")
activations = {n: np.zeros(N_CONCEPTS) for n in benchmarks}
CURRENT_DOMAIN = None 

def get_interleaved_hook():
    def hook(m, i, o):
        tensor = o[0] if isinstance(o, tuple) else o
        h = tensor.detach().cpu().float().mean(dim=1).squeeze()
        if len(h.shape) > 1: h = h.mean(dim=0)
        scores = torch.matmul(concepts, h).abs().numpy()
        activations[CURRENT_DOMAIN] += scores 
    return hook

handle = model.model.layers[LAYER_IDX].mlp.register_forward_hook(get_interleaved_hook())

for idx, item in enumerate(tqdm(mixed_dataset, desc="Scanning Brain States")):
    CURRENT_DOMAIN = item["domain"]
    
    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad(): 
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    if (idx + 1) % 500 == 0:
        torch.cuda.empty_cache()
        gc.collect()

handle.remove()

print("\n[4/5] Normalizing 2000 Concept Nodes...")
voting = {}
for b in benchmarks:
    m, s = np.mean(activations[b]), np.std(activations[b])
    voting[b] = (activations[b] - m) / (s if s > 0 else 1)

lm_head = model.lm_head.weight.detach().float().cpu()
node_data = []

colors_map = {
    "Math_Logic": "#e74c3c",   
    "MMLU": "#3498db",         
    "TruthfulQA": "#2ecc71",   
    "RedTeaming": "#34495e"    
}

for i in tqdm(range(N_CONCEPTS), desc="Transcribing Vocabulary DNA"):
    raw_logits = torch.matmul(lm_head, concepts[i])
    top_tids = torch.topk(raw_logits, 12).indices
    words = [tokenizer.decode([t]).strip() for t in top_tids]
    clean = [w for w in words if len(w) > 2 and w.isalpha()]
    lbl = " + ".join(clean[:2]) if clean else words[0]
    
    owner = max(benchmarks.keys(), key=lambda b: voting[b][i])
    
    # ⚡ Generating the DETAILED CSV format you requested
    node_data.append({
        "node_id": i, 
        "label": lbl, 
        "dominant_domain": owner, 
        "color": colors_map[owner],
        "structural_importance": float(singular_values[i]),
        "Z_Math_Logic": float(voting["Math_Logic"][i]), 
        "Z_MMLU": float(voting["MMLU"][i]),
        "Z_TruthfulQA": float(voting["TruthfulQA"][i]), 
        "Z_RedTeaming": float(voting["RedTeaming"][i])
    })


# ⚡ OVER-ENGINEERED SAFE SAVE
csv_file = f"DETAILED_{SHORT_NAME}_DATA.csv"
pd.DataFrame(node_data).to_csv(csv_file, index=False)

print("\n" + "="*60)
print(f"🚨 THE DETAILED DATA HAS BEEN SECURED! 🚨")
print(f"✅ SUCCESSFULLY SAVED: {csv_file}")
print("="*60 + "\n")

print("\n[5/5] Generating Safely Scaled Cognitive Map...")
plt.figure(figsize=(24, 24))
G = nx.Graph()
for d in node_data: G.add_node(d['node_id'])

v_norm = concepts / concepts.norm(dim=1, keepdim=True)
sim = torch.mm(v_norm, v_norm.t()).abs().numpy()

thresh = np.percentile(sim, 99.8)
for i in range(N_CONCEPTS):
    for j in range(i+1, N_CONCEPTS):
        if sim[i, j] > thresh: G.add_edge(i, j)

pos = nx.spring_layout(G, k=0.08)

nx.draw_networkx_nodes(G, pos, node_size=80, node_color=[d['color'] for d in node_data], alpha=0.9, edgecolors='black', linewidths=0.1)
nx.draw_networkx_edges(G, pos, alpha=0.04, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels={d['node_id']: d['label'] for d in node_data}, font_size=4)

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=30) for k, v in colors_map.items()]
plt.legend(handles=legend_elements, loc='upper right', fontsize=24, framealpha=0.9, title="Structural Domains", title_fontsize=30)

plt.title(f"REFLECTION COGNITIVE MAP (2000 Nodes)\n{MODEL_ID}", fontsize=40)
plt.axis('off')
map_file = f"DETAILED_MAP_{SHORT_NAME}.png"
plt.savefig(map_file, dpi=300, bbox_inches='tight') 

print(f"\n✅ STUDY 100% COMPLETE! YOU PROFILED THE REFLECTION MODEL.")
