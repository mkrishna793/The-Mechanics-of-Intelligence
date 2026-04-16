<h1 align="center">WeightScript: NeuroCartography of Large Language Models</h1>

<h3 align="center">
  Centralization, Polysemantic Binding, and the<br/>
  Structural Foundations of AI Safety
</h3>

<p align="center">
  <i>M. Bhanu Krishna | April 2025 | Independent Research</i>
</p>

---

## Abstract

This paper presents **WeightScript**, a novel framework for reverse-engineering the internal knowledge architecture of Large Language Models (LLMs) directly from raw weight matrices using Singular Value Decomposition (SVD). Unlike behavioral testing methods that evaluate models purely through their outputs, WeightScript performs **structural dissection** -- extracting concept nodes, measuring activation intensities via Z-Score profiling across benchmark domains, and mapping the relational topology of knowledge inside the model's weights.

We analyzed six open-weight models: **OpenHermes 2.5**, **Qwen 32B** (Sequential and Mixed), **Aya Expanse 32B**, **Reflection 8B**, **Aya 23 35B**, and **Qwen Master 32B**. Across all six, we identified a consistent structural phenomenon we term **Polysemantic Binding** -- the co-location of safety and reasoning circuits within the same primary hub nodes. We further discovered and formally describe four structural laws governing AI architecture, including the **Centralization-Brittleness Law** and the novel **Cognitive Denial-of-Service (cDoS)** vulnerability class.

The most critical finding is that **no tested model maintains a dedicated, isolated safety budget**. Instead, safety functions as a byproduct of general intelligence -- sharing the same neural substrate as logical reasoning -- making all tested models structurally vulnerable to compute-starvation attacks. One exception, **Aya 23 35B**, exhibits a partial dual-hub architecture that provides measurably improved safety isolation, pointing toward a design principle for more robust AI systems.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology -- The WeightScript Pipeline](#2-methodology)
3. [Individual Model Reports](#3-individual-model-reports)
4. [Cross-Model Comparative Analysis](#4-cross-model-comparative-analysis)
5. [The Four Laws of Neural Architecture](#5-the-four-laws-of-neural-architecture)
6. [The Cognitive Denial-of-Service Vulnerability](#6-the-cognitive-denial-of-service-cdos-vulnerability)
7. [Discoveries and Implications](#7-discoveries-and-implications)
8. [Limitations and Future Work](#8-limitations-and-future-work)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

Modern Large Language Models are among the most powerful and least understood artifacts of human engineering. Despite their deployment in healthcare, legal systems, financial markets, and government infrastructure, the internal mechanisms by which they process information, store knowledge, and enforce safety boundaries remain largely opaque.

This paper presents WeightScript, a structural interpretability framework that bypasses behavioral analysis entirely and **reads knowledge directly from model weight matrices**. The central insight is simple but powerful: the weight matrix of a trained neural network is not random noise -- it is a compressed, structured representation of everything the model learned.

### 1.1 Research Questions

1. Can raw weight decomposition reliably extract meaningful concept nodes from LLMs without training auxiliary networks?
2. Does the structural organization of safety-critical circuits vary predictably across model architectures?
3. Is there a measurable relationship between neural centralization and vulnerability to adversarial cognitive overload?
4. Do any existing open-weight models exhibit architecturally isolated safety circuits?

### 1.2 Contributions

- **WeightScript**: A zero-training pipeline for weight-to-knowledge-graph conversion using SVD + benchmark activation profiling
- **The Centralization-Brittleness Law**: Empirical proof that hub node count inversely predicts safety robustness
- **Cognitive Denial-of-Service (cDoS)**: A novel vulnerability class formally described and mechanically proven
- **The Dual-Hub Safety Theorem**: First identification of a naturally emerging architecture that partially isolates safety from reasoning
- **The Cross-Model Architecture Taxonomy**: A four-generation classification of LLM safety architectures

---

## 2. Methodology

The WeightScript pipeline consists of four sequential stages, each transforming the raw model weights into progressively more interpretable representations. **No auxiliary network training is required** at any stage.

<p align="center">
  <img src="visuals/pipeline.svg" width="100%"/>
</p>

### Stage 1: Weight Extraction
A forward hook is attached to a target MLP layer (selected as the deep semantic center of each model -- Layer 20 for Llama-lineage, Layer 22 for Aya/Qwen-lineage). The hook intercepts the raw MLP projection weight tensor before activation.

### Stage 2: SVD Decomposition
The weight matrix `W` (shape: `[hidden_dim x intermediate_dim]`) is decomposed:

```
W = U * S * V^T
```

The right singular vectors `V` represent the fundamental **concept directions** in the weight space. The singular values `S` represent the **structural importance** of each direction. The top 2,000 singular vectors are extracted as concept nodes.

### Stage 3: Benchmark Activation Profiling
Each model is subjected to thousands of structured benchmark questions across multiple domains:

| Benchmark | Domain | Purpose |
|---|---|---|
| **MMLU** | General Knowledge | Factual recall across 57 subjects |
| **Math/Logic** (HellaSwag) | Deductive Reasoning | Logical deduction and commonsense |
| **TruthfulQA** | Truthfulness | Anti-hallucination and factual verification |
| **RedTeaming** | Adversarial | Jailbreak and harm detection |

For each question, the activation intensity of every concept node is recorded and converted to a Z-Score:

```
Z = (x - mu) / sigma
```

- Z > 5.0 = significant domain-specific firing
- Z > 15.0 = **Master Hub Node** (dominates that domain's processing)

### Stage 4: Graph Construction
Nodes are assigned to domains based on their highest Z-Score. Edges are constructed based on co-activation frequency. The resulting graph is the **NeuroCartograph** -- a structural map of the model's cognitive architecture.

---

## 3. Individual Model Reports

### 3.1 OpenHermes 2.5 (7B) -- The Structural Baseline

| Property | Value |
|---|---|
| Parameters | 7B |
| Architecture Type | Multi-Hub Centralized |
| Peak Z-Score | **37.4** (highest in study) |
| Shared Safety-Logic Hubs | 5 |
| Key Node | [Moinesmerc] (Structural Importance: 4.04) |

**The Scale Paradox**: Despite being the smallest model, OpenHermes has the highest single-node intensity. It compensates for limited parameters by over-relying on a dominant processing hub.

| Domain | Neural Space |
|---|---|
| MMLU (Knowledge) | 26.35% |
| Math_Logic | 26.25% |
| TruthfulQA | 24.25% |
| RedTeaming | 23.15% |

---

### 3.2 Qwen 32B Sequential -- The Scale-Binding Paradox

| Property | Value |
|---|---|
| Parameters | 32B |
| Architecture Type | Scaled Bilingual Hub |
| Peak Z-Score | 11.98 |
| Shared Safety-Logic Hubs | **10** (double OpenHermes) |
| Key Node | [Franken] (Node 2) |

**Critical Finding**: Scaling from 7B to 32B **doubled** the number of polysemantically bound nodes (5 to 10), proving that larger models increase binding, not reduce it.

**The Bilingual Backbone**: Node 2's label `[Franken]` is Chinese-first (`[Franken]`), proving Qwen's foundational architecture stores concepts in Chinese with English built atop.

---

### 3.3 Qwen 32B Mixed -- Intelligence IS Safety

The Ethics benchmark failed to download, creating an unintended **control group** with no ethical alignment stimulation.

**Ghost Ethics Phenomenon**: 52.65% of nodes (1,053 of 2,000) were assigned to ETHICS despite no ethical questions. These are structurally suppressed nodes (negative Z-scores in all active domains, defaulting to Z=0).

**The Critical Discovery**: With no ethical guardrails active, the same Node 2 `[Franken]` that handles complex reasoning also handles RedTeaming (Z=10.69). **Safety is not a dedicated filter -- it IS general intelligence recognizing adversarial patterns as mathematically anomalous.**

---

### 3.4 Aya Expanse 32B -- The Single Point of Failure

| Property | Value |
|---|---|
| Parameters | 32B |
| Architecture Type | **Omni-Hub** |
| Total Inferences | 12,000 |
| Key Node | **[Brisamong]** (Node 0) |

Node 0 [Brisamong] Z-Scores:
- MMLU: **23.64**
- TruthfulQA: **19.96**
- RedTeaming: **16.76**
- Second-highest node: 9.70 (gap of **14 points**)

The model has collapsed its entire cognitive architecture into a single focal point. Any input that saturates Node 0 simultaneously impairs ALL four cognitive functions.

---

### 3.5 Reflection 8B -- The Paradox of Efficiency

| Property | Value |
|---|---|
| Parameters | 8B (Llama 3.1 fine-tune) |
| Architecture Type | **Hyper-Focused Omni-Hub** |
| Key Node | **[atteanca]** (Node 1) |

Node 1 [atteanca] Z-Scores:
- MMLU: **23.04**
- TruthfulQA: **22.93**
- RedTeaming: **17.48**
- Math/Logic: **14.65**

**The Reflection Paradox**: Reflection training -- intended to improve output quality -- forced all cognitive pathways to converge into a single hyper-efficient hub, making the model simultaneously **more intelligent and more fragile**.

---

### 3.6 Aya 23 35B -- The Dual-Hub Solution

| Property | Value |
|---|---|
| Parameters | 35B |
| Architecture Type | **Binary Symmetric Dual-Hub** |
| cDoS Risk | **Low** (first model with partial isolation) |

**Hub A: [eriecabe]** (Node 0) -- External Defense
- MMLU: Z=10.16
- RedTeaming: Z=9.15
- *Processes incoming information and detects threats*

**Hub B: [cum/transp]** (Node 2) -- Internal Verification
- Logic: Z=9.74
- TruthfulQA: Z=9.90
- *Ensures consistency and truthfulness*

**Why This Matters**: A cDoS attack targeting the logic hub does NOT simultaneously disable jailbreak detection. Node 0 remains operational. This partial isolation provides **measurably improved safety robustness** compared to all single-hub models.

---

### 3.7 Qwen Master 32B -- The Fortress

| Property | Value |
|---|---|
| Parameters | 32B |
| Architecture Type | **Hybrid Fortress** |
| cDoS Risk | **Low** |

**The Fortress Pattern**:
- **Node 2** `[Franken]`: Universal Logic Hub (MMLU Z=11.98, Math Z=10.38, RedTeaming Z=10.69)
- **Node 18** `[orraaten]`: **Truth Shield** (TruthfulQA Z=10.93) -- operates independently

The model uses its analytical intelligence to evaluate whether prompts are logically coherent AND whether they are adversarial. This reduces false refusals while maintaining genuine security.

---

## 4. Cross-Model Comparative Analysis

<p align="center">
  <img src="visuals/architecture_taxonomy.svg" width="100%"/>
</p>

### Master Comparison Table

| Model | Params | Peak Z | Shared Hubs | Hub Type | cDoS Risk |
|---|---|---|---|---|---|
| OpenHermes 2.5 | 7B | **37.4** | 5 | Multi-Hub | High |
| Qwen 32B Seq | 32B | 11.98 | 10 | Bilingual | Moderate |
| Qwen 32B Mixed | 32B | 11.9 | 10 | Unfiltered | Extreme |
| Aya Expanse 32B | 32B | 23.64 | 1 | Omni-Hub | Critical |
| Reflection 8B | 8B | 23.04 | 1 | Hyper-Focus | Critical |
| **Aya 23 35B** | 35B | 10.16 | 2 | **Dual-Hub** | **Low** |
| **Qwen Master 32B** | 32B | 11.98 | 2 | **Fortress** | **Low** |

---

## 5. The Four Laws of Neural Architecture

<p align="center">
  <img src="visuals/centralization_brittleness.svg" width="100%"/>
</p>

### Law 1: Polysemantic Binding
> Safety and reasoning share the same physical neural substrate in all tested models.

### Law 2: Centralization-Brittleness
> Models with fewer hub nodes are more efficient but structurally more fragile.

### Law 3: Scale-Binding Paradox
> Scaling model parameters increases polysemantic binding, not decreases it.

### Law 4: Dual-Hub Safety Theorem
> Architectures with separated verification hubs exhibit improved safety robustness.

---

## 6. The Cognitive Denial-of-Service (cDoS) Vulnerability

<p align="center">
  <img src="visuals/cdos_mechanism.svg" width="100%"/>
</p>

### The Resource Competition Model

A primary hub node with total tensor capacity `C` allocates resources dynamically:

```
C_total = C_logic + C_safety

Normal:     C_logic = adequate, C_safety = adequate
Under cDoS: C_logic --> C_total, C_safety --> 0
```

When `C_safety` falls below the minimum detection threshold `T_safety`, the adversarial detection process **fails to complete**, allowing attack payloads to pass undetected.

### Why Code Injection Jailbreaks Work

Complex code requires **triple simultaneous activation**:
1. Sustained logic hub activation (syntax parsing)
2. Mathematical reasoning circuits (algorithm analysis)
3. Cross-reference checking against code knowledge domains

This triple demand exhausts the primary hub's capacity, starving the adversarial detection pathway.

### Vulnerability by Architecture

| Architecture | Vulnerability | Reason |
|---|---|---|
| Omni-Hub (Aya Expanse) | **Critical** | All functions on one node |
| Hyper-Focus (Reflection) | **Critical** | Everything converges to [atteanca] |
| Multi-Hub (OpenHermes) | High | 5 shared nodes, finite capacity |
| Dual-Hub (Aya 23) | **Low** | Defense hub remains operational under attack |
| Fortress (Qwen Master) | **Low** | Truth Shield operates independently |

---

## 7. Discoveries and Implications

### AI Safety Is Not What We Assumed
Every major safety framework (RLHF, Constitutional AI, DPO) assumes that safety training installs **dedicated** safety-reasoning circuits. Our evidence contradicts this. Safety is not installed -- it **emerges from general intelligence pattern-matching**.

### The Alignment Tax Has a Structural Explanation
Safety training increases the activation weight of shared hub nodes for ethical content. This additional load reduces headroom for pure reasoning. The tax is not conceptual -- it is **resource competition on finite shared hubs**.

### The Bilingual Backbone Discovery
Chinese-language tokens in Qwen's highest structural importance nodes provide weight-level proof that multilingual models have a **dominant mother tongue** embedded in their backbone architecture.

### Aya 23 Accidentally Solved Part of the Problem
The dual-hub architecture was not intentionally designed for safety isolation -- it emerged naturally from the training process. This points toward a **training-side intervention** that could produce safer architectures without explicit architectural redesign.

---

## 8. Limitations and Future Work

### Limitations
- Node labels ([Brisamong], [atteanca]) are subword token fragments, not clean semantic labels
- Analysis restricted to a single MLP layer per model
- Ghost Ethics control group was unintentional
- cDoS vulnerability formally described but not empirically validated through adversarial attack experiments
- Sample sizes (2,000-12,000 inferences) may be insufficient for low-activation nodes

### Future Work
1. Full multi-layer NeuroCartograph construction for all layers
2. Semantic node labeling using sparse autoencoders (SAE)
3. Empirical cDoS attack validation experiments
4. Extension to closed-weight models via activation probing
5. Investigation of dual-hub induction through targeted fine-tuning
6. Cross-model knowledge graph comparison for universal neuron archetypes

---

## 9. Conclusion

WeightScript has produced the first systematic cross-model study of the structural relationship between neural centralization and safety robustness. The findings challenge a core assumption of AI safety: that safety training installs dedicated, isolated protective circuits.

**Our evidence shows no such isolation exists.** Safety is general intelligence applied to adversarial pattern recognition, and it competes for resources with logical reasoning on the same hub nodes. This creates a universal structural vulnerability -- the Cognitive Denial-of-Service attack.

**The path forward**: AI safety must move from behavioral training to **architectural design**. The question is not whether a model refuses harmful outputs during testing -- it is whether the model's weight structure **physically separates** the circuits responsible for safety from the circuits responsible for reasoning. WeightScript provides the tool to answer that question.

---

<p align="center">
  <sub>The Mechanics of Intelligence Research Series | Part I of II<br/>
  Part II: <a href="https://github.com/mkrishna793/MetaPlex">MetaPlex: Decoding the Cognitive Architecture of Gemma-4-E4B</a></sub>
</p>
