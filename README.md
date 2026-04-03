# 🧬 The Mechanics of Intelligence: WeightScript Framework
**Structural NeuroCartography and the Mathematics of Neural Centralization**

![Master Map](plots/MAP_QWEN_MASTER_32B.png)

## 📖 Overview
**The Mechanics of Intelligence** is a structural interpretability project dedicated to reverse-engineering the internal knowledge architecture of Large Language Models (LLMs). Utilizing the **WeightScript** framework, we bypass behavioral testing and read mathematical intent directly from the model's frozen weight matrices.

By performing **Singular Value Decomposition (SVD)** on deep semantic layers and profiling activation intensities (**Z-Scores**) across thousands of benchmarks, we have produced the first systematic cross-model structural comparison of LLM safety and reasoning.

## 🔬 Core Discoveries
Our research across six frontier open-weight models (8B to 35B) has yielded three foundational discoveries:

1.  **Polysemantic Binding:** We have proven that modern LLMs do not maintain dedicated safety circuits. Instead, safety and logic are co-located on the same physical neural substrate.
2.  **The Centralization-Brittleness Law:** We established that models with fewer "Hub Nodes" (high neural centralization) are measurably more efficient but structurally more fragile.
3.  **Cognitive Denial-of-Service (cDoS):** We formally described a novel vulnerability class where complex logical puzzles are used to saturate a model's primary hubs, "starving" the safety detection neurons and allowing adversarial payloads to pass undetected.

---

## 🗺️ Master NeuroCartograph Gallery
The following maps represent the structural "continents" of intelligence inside each model. Each color represents a specialized cognitive domain (Math, Ethics, Truthfulness, Adversarial Detection).

| **Aya Expanse 32B (The Omni-Node)** | **Qwen Master 32B (The Fortress)** |
|:---:|:---:|
| ![Aya Expanse](plots/MAP_AYA_EXPANSE_32B.png) | ![Qwen Master](plots/MAP_QWEN_MASTER_32B.png) |
| *Extreme Centralization in [Brisamong]* | *Symmetric Defense & Truth Shield* |

| **Reflection 8B (The Paradox)** | **Aya 23 35B (The Dual-Hub)** |
|:---:|:---:|
| ![Reflection](plots/MAP_REFLECTION_8B.png) | ![Aya 23](plots/MAP_AYA_23_35B.png) |
| *Hyper-Focus into [atteanca]* | *Separated Verification and Logic* |

---

## 📈 Cross-Model Taxonomy
Based on WeightScript analysis, we have classified LLMs into four architectural generations of safety robustness:

| Model | Architectural Type | Key Discovery | Z-Score (Peak) | cDoS Risk |
|---|---|---|---|---|
| **OpenHermes 2.5** | Multi-Hub Baseline | Moderate Polysemantic Binding | 37.4 | High |
| **Qwen Mixed 32B** | Unfiltered Control | Intelligence AS Safety | 11.9 | Extreme |
| **Aya Expanse 32B** | Omni-Hub | The [Brisamong] Single Point of Failure | 23.6 | Critical |
| **Reflection 8B** | Hyper-Focused | The Efficiency-Safety Tradeoff | 23.0 | Critical |
| **Aya 23 35B** | Dual-Hub Symmetric | Separated Verification and Logic | 10.1 | Low |
| **Qwen Master 32B** | Hybrid Fortress | The [orraaten] Truth Shield | 11.9 | Low |

---

## 📂 Repository Structure
- **[/reports](reports/)**: Detailed "Ultra-Deep Discovery" reports for each model.
- **[/src](src/)**: The WeightScript Python pipeline (SVD, Probing, Synthesis).
- **[/plots](plots/)**: High-Resolution NeuroCartographs and distribution charts.
- **[/paper](paper/)**: The original WeightScript Research Paper.
- **[/data](data/)**: Structural importance and activation CSV datasets.

---

## 🚀 Getting Started
To run the WeightScript pipeline on a new model:
1.  Configure your HuggingFace token in `src/universal_cartography.py`.
2.  Set your target `MODEL_ID` and `LAYER_IDX`.
3.  Run the pipeline:
    ```bash
    python src/universal_cartography.py
    ```

## 📜 Official Paper
For the full mathematical proof and experimental data, please read the primary research document: [WeightScript: NeuroCartography of Large Language Models](paper/WeightScript_NeuroCartography.md).

---
*Created by mkrishna793. Part of the Mechanics of Intelligence Research Series.*
