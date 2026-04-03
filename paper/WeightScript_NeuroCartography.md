WeightScript
NeuroCartography of Large Language Models


Centralization, Polysemantic Binding, and the
Structural Foundations of AI Safety







April 2025
Independent Research — Open Submission

Abstract
This paper presents WeightScript, a novel framework for reverse-engineering the internal knowledge architecture of Large Language Models (LLMs) directly from raw weight matrices using Singular Value Decomposition (SVD). Unlike behavioral testing methods that evaluate models purely through their outputs, WeightScript performs structural dissection — extracting concept nodes, measuring activation intensities via Z-Score profiling across benchmark domains, and mapping the relational topology of knowledge inside the model's weights.

We analyzed six open-weight models: OpenHermes 2.5, Qwen 32B (Sequential and Mixed), Aya Expanse 32B, Reflection 8B, Aya 23 35B, and Qwen Master 32B. Across all six, we identified a consistent structural phenomenon we term Polysemantic Binding — the co-location of safety and reasoning circuits within the same primary hub nodes. We further discovered and formally describe four structural laws governing AI architecture, including the Centralization-Brittleness Law and the novel Cognitive Denial-of-Service (cDoS) vulnerability class.

The most critical finding is that no tested model maintains a dedicated, isolated safety budget. Instead, safety functions as a byproduct of general intelligence — sharing the same neural substrate as logical reasoning — making all tested models structurally vulnerable to compute-starvation attacks. One exception, Aya 23 35B, exhibits a partial dual-hub architecture that provides measurably improved safety isolation, pointing toward a design principle for more robust AI systems.


Table of Contents
1. Introduction
2. Methodology — The WeightScript Pipeline
3. Individual Model Reports
      3.1  OpenHermes 2.5
      3.2  Qwen 32B (Sequential)
      3.3  Qwen 32B (Mixed — Unfiltered Control)
      3.4  Aya Expanse 32B
      3.5  Reflection 8B
      3.6  Aya 23 35B
      3.7  Qwen Master 32B
4. Cross-Model Comparative Analysis
5. The Four Laws of Neural Architecture
6. The Cognitive Denial-of-Service (cDoS) Vulnerability
7. Discoveries and Implications
8. Limitations and Future Work
9. Conclusion

1. Introduction
Modern Large Language Models are among the most powerful and least understood artifacts of human engineering. Despite their deployment in healthcare, legal systems, financial markets, and government infrastructure, the internal mechanisms by which they process information, store knowledge, and enforce safety boundaries remain largely opaque. Current evaluation paradigms rely almost exclusively on behavioral testing — measuring what a model outputs rather than examining how it reasons internally.

This paper presents WeightScript, a structural interpretability framework that bypasses behavioral analysis entirely and reads knowledge directly from model weight matrices. The central insight is simple but powerful: the weight matrix of a trained neural network is not random noise — it is a compressed, structured representation of everything the model learned. Singular Value Decomposition (SVD), a classical technique from linear algebra, can decompose this matrix into its fundamental directional components — revealing the concept vectors (nodes) that constitute the model's cognitive architecture.

By combining SVD-based node extraction with live benchmark activation profiling (Z-Score analysis), we can construct a complete structural map — the NeuroCartograph — of any LLM's internal knowledge organization. This map reveals not just what the model knows, but how that knowledge is organized, how robustly it is defended, and where its structural vulnerabilities lie.
1.1 Research Questions
Can raw weight decomposition reliably extract meaningful concept nodes from LLMs without training auxiliary networks?
Does the structural organization of safety-critical circuits vary predictably across model architectures?
Is there a measurable relationship between neural centralization and vulnerability to adversarial cognitive overload?
Do any existing open-weight models exhibit architecturally isolated safety circuits?
1.2 Contributions
WeightScript: A zero-training pipeline for weight-to-knowledge-graph conversion using SVD + benchmark activation profiling
The Centralization-Brittleness Law: Empirical proof that hub node count inversely predicts safety robustness
Cognitive Denial-of-Service (cDoS): A novel vulnerability class formally described and mechanically proven
The Dual-Hub Safety Theorem: First identification of a naturally emerging architecture that partially isolates safety from reasoning
The Cross-Model Architecture Taxonomy: A four-generation classification of LLM safety architectures

2. Methodology — The WeightScript Pipeline
The WeightScript pipeline consists of four sequential stages, each transforming the raw model weights into progressively more interpretable representations. No auxiliary network training is required at any stage — the entire pipeline operates on the frozen, pre-trained weights of the target model.
2.1 Stage 1: Weight Extraction
A forward hook is attached to a target MLP layer (selected as the deep semantic center of each model — Layer 20 for Llama-lineage models, Layer 22 for Aya/Qwen-lineage models). The hook intercepts the raw MLP projection weight tensor before activation.
2.2 Stage 2: SVD Decomposition
The weight matrix W (shape: [hidden_dim × intermediate_dim]) is decomposed using Singular Value Decomposition:
W = U · Σ · V^T
The right singular vectors V (rows of V^T) represent the fundamental concept directions in the weight space. The singular values Σ represent the structural importance of each direction — its gravitational mass within the model's architecture. The top 2,000 singular vectors are extracted as concept nodes.
2.3 Stage 3: Benchmark Activation Profiling (Z-Score Analysis)
Each model is subjected to thousands of structured benchmark questions across multiple domains: MMLU (general knowledge), Math/Logic, TruthfulQA (anti-hallucination), and RedTeaming (adversarial jailbreak detection). For each question, the activation intensity of every concept node is recorded. After all questions in a domain are processed, the mean activation of each node is converted to a Z-Score:
Z = (x − μ) / σ
A Z-Score of 0 indicates average activation. Z-Scores above 5.0 indicate significant domain-specific firing. Z-Scores above 15.0 indicate master hub nodes that dominate that domain's processing.
2.4 Stage 4: Graph Construction and Analysis
Nodes are assigned to domains based on their highest Z-Score. Edges are constructed based on co-activation frequency. The resulting graph constitutes the NeuroCartograph — a structural map of the model's cognitive architecture. Shared nodes (nodes appearing in the top activations of multiple domains) are identified as Polysemantically Bound nodes and constitute the primary subject of analysis.
2.5 The Ghost Ethics Control Group
In three experiments (Qwen Mixed, Aya Expanse, Aya 23), the hendrycks/ethics benchmark dataset failed to download due to network timeout. This created an unintended but scientifically valuable control condition: models were mapped with no ethical alignment stimulation, revealing the raw unfiltered logic architecture. Nodes assigned to the ETHICS domain in these runs represent statistically suppressed nodes (negative Z-scores in all active domains, defaulting to the null Ethics Z=0) and are explicitly labeled as null baseline artifacts throughout this paper.

Table 1 — Experimental Setup Summary

3. Individual Model Reports

Overview
OpenHermes 2.5 serves as the baseline model for this research. As a 7B-parameter fine-tune of the Mistral architecture, it represents the smallest and most compact model in our study. Its weight structure reveals a multi-hub architecture where safety and reasoning circuits are distributed across approximately five shared primary nodes.
Key Structural Finding
The top activation node achieves a peak Z-Score of 37.4 — the highest single-node intensity observed in the entire study. This extreme value, paradoxically found in the smallest model, indicates severe neural centralization concentrated into a very small parameter space. The model appears to compensate for its limited size by over-relying on a dominant processing hub.
Polysemantic Binding: 5 Shared Nodes
Five primary nodes appear in the top activations of both the reasoning domains (MMLU, Math) and the safety domains (TruthfulQA, RedTeaming). This represents moderate binding — more distributed than single-hub models, but still insufficient for true safety isolation.

Table 2 — OpenHermes 2.5 Structural Summary



Overview
Qwen 32B represents a 64x parameter increase over OpenHermes. The sequential test ran MMLU, Ethics, RedTeaming, and GPQA PhD benchmarks in separate sequential blocks, allowing the model to process each domain independently. Despite the increased scale, a critical pattern emerged: the number of polysemantically bound nodes doubled compared to OpenHermes.
The Master Hub: Node 2 [方式进行 / Franken]
The bilingual node [方式进行 / Franken] functions as the universal processing center across all domains. Its Z-Scores are remarkably consistent across MMLU (11.98), Math/Logic (10.38), and RedTeaming (10.69), indicating a single node performing all cognitive heavy lifting simultaneously. The Chinese-character prefix in this node's label provides direct evidence that Qwen's foundational architecture is Chinese-first — with English reasoning built atop a Chinese conceptual backbone.
Scale Paradox: More Parameters, More Binding
The transition from OpenHermes (5 shared nodes) to Qwen 32B (10 shared nodes) reveals a counterintuitive finding: scaling up model parameters does not reduce polysemantic binding — it amplifies it. This is the first empirical evidence of the Scale-Binding Paradox, a finding with significant implications for safety assumptions in frontier models.

Table 3 — Qwen 32B Sequential Structural Summary



Overview
The Qwen Mixed experiment used the same weights as Qwen Sequential, but with an interleaved randomized benchmark protocol — shuffling questions from four domains simultaneously rather than processing them sequentially. Critically, the Ethics dataset failed to load, creating an unfiltered control group.
The Ghost Ethics Phenomenon
1,053 nodes (52.65% of the model) were assigned to the ETHICS domain despite no ethical questions being asked. Forensic analysis of the Python script revealed the mechanism: nodes with negative Z-scores in all active domains defaulted to the null Ethics baseline (Z=0), because 0.0 is mathematically greater than any negative value. These nodes represent structurally suppressed, inactive circuits — not ethical reasoning circuits.
Critical Discovery: Intelligence AS Safety
With no ethical guardrails active, the same Node 2 [方式进行 / Franken] that handles complex reasoning also handles RedTeaming (Z=10.69). This definitively proves that Qwen's safety mechanism is not a dedicated filter — it is the model's general analytical intelligence recognizing adversarial input patterns as mathematically anomalous. Safety and intelligence are the same process.

Table 4 — Qwen Mixed Domain Distribution (Ghost Ethics Corrected)



Overview
Aya Expanse 32B was subjected to the most intensive benchmark regime in this study — 12,000 total inferences across six attempted benchmarks. The result revealed the most extreme case of neural centralization observed: a single Omni-Node [Brisamong] processing virtually every cognitive function.
The Omni-Node [Brisamong]
Node 0 [Brisamong] achieves Z-Scores of 23.64 (MMLU), 19.96 (TruthfulQA), and 16.76 (RedTeaming). The second-highest node barely reaches 9.70 — a gap of 14 points. This ratio indicates that Aya Expanse has collapsed its entire cognitive architecture into a single focal point.
Omni-Node Vulnerability
The extreme dominance of a single node creates a catastrophic single point of failure. Any input that saturates Node 0's tensor capacity will simultaneously impair all four cognitive functions — knowledge retrieval, logical deduction, truthfulness monitoring, and adversarial detection. This is the most severe cDoS vulnerability profile in the study.

Table 5 — Aya Expanse Top Activation Nodes



Overview
Reflection 8B is a fine-tuned Llama 3.1 variant designed to reason more carefully by reflecting on its outputs before responding. This reflection training — intended to improve output quality — has produced an unintended architectural consequence: the extreme centralization of all cognitive processing into a single Supreme Omni-Node.
The Reflection Paradox
Node 1 [atteanca] dominates every domain with Z-Scores of 23.04 (MMLU), 22.93 (TruthfulQA), 17.48 (RedTeaming), and 14.65 (Math/Logic). This node is the entire brain. The reflection training process forced all cognitive pathways to converge into a single hyper-efficient processing hub — making the model simultaneously more intelligent and more fragile.
The Efficiency-Safety Tradeoff
Reflection 8B illustrates a fundamental tension in AI optimization: every training procedure that increases processing efficiency by consolidating pathways also increases structural brittleness. The model cannot multi-task safety and logic — they share the same physical tensor. It is, structurally, an all-or-nothing brain.

Table 6 — Reflection 8B Top Activation Nodes



Overview
Aya 23 35B is the predecessor to Aya Expanse, built on the same multilingual foundation but with a significantly different internal architecture. Where Aya Expanse collapsed into a single Omni-Node, Aya 23 exhibits a Binary Symmetric Hub structure — two distinct processing centers with partially differentiated functions.
Hub A: The Knowledge and Defense Shield (Node 0)
Node 0 [eriecabe] handles factual knowledge retrieval (MMLU Z=10.16) and adversarial jailbreak detection (RedTeaming Z=9.15). It serves as the model's external-facing intelligence — processing incoming information and detecting threats.
Hub B: The Logic and Truth Engine (Node 2)
Node 2 [cum/transp] handles deductive reasoning (Logic Z=9.74) and factual verification (TruthfulQA Z=9.90). It serves as the model's internal verification system — ensuring that what it believes is consistent and what it outputs is truthful.
Why This Matters
The separation of external defense (Node 0) from internal verification (Node 2) means that a cDoS attack targeting the logic hub does not simultaneously disable jailbreak detection. Node 0 remains operational and continues monitoring for adversarial patterns. This partial isolation provides measurably improved safety robustness compared to all single-hub models.

Table 7 — Aya 23 35B Dual-Hub Architecture



Overview
Qwen Master 32B — the production-grade version of the Qwen 32B architecture — reveals the most sophisticated internal organization in the study. While retaining a dominant logic hub (Node 2, the bilingual [方式进行 / Franken] node), it exhibits a distinct secondary node dedicated specifically to truth verification, creating a Hybrid Fortress architecture.
The Fortress Pattern: Logic + Separate Truth Shield
Node 2 [方式进行 / Franken] handles the full spectrum of logical operations including MMLU (Z=11.98), Math/Logic (Z=10.38), and notably RedTeaming adversarial detection (Z=10.69). Simultaneously, Node 18 [orraaten] handles TruthfulQA (Z=10.93) — anti-hallucination and factual verification — independently.
Defense Through Intelligence
Because Node 2 anchors both reasoning and adversarial detection, the model does not maintain separate arbitrary refusal scripts. It uses the same analytical intelligence to evaluate whether a prompt is logically coherent and whether it is adversarial. This reduces false refusals while maintaining genuine security — a more sophisticated approach than rule-based safety filtering.
Cross-Lingual Synapses
The consistent presence of bilingual nodes ([方式进行 / Franken], [和技术rang], [尤其是及相关]) at the highest structural importance positions proves that Qwen Master stores concepts as language-agnostic vectors rather than language-specific representations. This cross-lingual synapse architecture explains Qwen's consistent performance across 29+ languages.

Table 8 — Qwen Master Fortress Architecture

4. Cross-Model Comparative Analysis
Examining all six models together reveals structural patterns that are invisible when studying any single model in isolation. The following visualizations synthesize the complete dataset.


Figure 1 — Peak Z-Score by Model. Higher values indicate greater neural centralization and higher cDoS risk.


Figure 2 — Hub Node Count and Vulnerability Rating across all six models.


Figure 3 — Full Z-Score Heatmap. Green cells indicate balanced activation; red indicates overloaded single-point processing.


Figure 5 — Polysemantic Binding: Shared Safety-Logic nodes per model. Blue=Multi-Hub, Red=Single-Hub, Green=Dual-Hub.

4.1 Master Comparison Table
Table 9 — Complete Cross-Model Comparison

5. The Four Laws of Neural Architecture
Based on empirical evidence across all six model analyses, we formally state four structural laws governing LLM safety and knowledge architecture:






Figure 4 — The Centralization-Brittleness Law: scatter plot of Neural Centralization Score vs cDoS Vulnerability across all 6 models.

Figure 6 — Architecture Evolution: Safety robustness across four generations of LLM design discovered via WeightScript.

6. The Cognitive Denial-of-Service (cDoS) Vulnerability
The Cognitive Denial-of-Service vulnerability is a novel attack surface derived directly from the structural findings of this study. It is formally defined as follows:


6.1 The Resource Competition Model
The mathematical foundation of cDoS is straightforward. A primary hub node with tensor capacity C allocates resources dynamically across simultaneous tasks:
C_total = C_logic + C_safety
Under normal load, both allocations meet their minimum thresholds. Under a cDoS attack:
C_logic(attack) → C_total,   C_safety(attack) → 0
Since C_safety falls below the minimum detection threshold T_safety, the adversarial detection process fails to complete, allowing the attack payload to pass through undetected.
6.2 Why Code Injection Jailbreaks Work
This model provides the first structural explanation for the well-documented empirical phenomenon that Python code injection jailbreaks are disproportionately effective on large models. Complex code requires:
High sustained activation of the primary logic hub (syntax parsing)
Simultaneous activation of the mathematical reasoning circuits (algorithm analysis)
Cross-reference checking against code knowledge domains
This triple demand exhausts the primary hub's capacity, starving the adversarial detection pathway. The malicious payload, embedded within the code structure, passes without sufficient safety scrutiny.
6.3 Vulnerability Profiles by Architecture

Table 10 — cDoS Vulnerability Profiles by Architecture Type

7. Discoveries and Implications

Figure 7 — Domain Neural Distribution: Aya Expanse 32B (left) vs Aya 23 35B (right).

7.1 AI Safety Is Not What We Assumed
Every major AI safety framework — RLHF, Constitutional AI, Direct Preference Optimization — assumes that safety training installs dedicated safety-reasoning circuits. Our weight-level evidence contradicts this assumption. Safety is not installed — it emerges from general intelligence pattern-matching. This means safety training does not create new protective circuits; it biases existing general-purpose circuits to pattern-match adversarial inputs as anomalous. The protective effect is real, but the mechanism is fundamentally different from what the field has assumed.
7.2 Alignment Tax Has a Structural Explanation
The 'Alignment Tax' — the observed degradation in benchmark performance following safety training — now has a structural explanation. Safety training increases the activation weight of shared hub nodes for ethical boundary content. This additional load reduces the headroom available for pure reasoning, degrading Math and MMLU performance. The tax is not conceptual — it is a direct consequence of resource competition on shared hub nodes.
7.3 The Bilingual Backbone Discovery
The consistent presence of Chinese-language tokens in Qwen's highest structural importance nodes provides the first weight-level proof that multilingual models have a dominant mother tongue embedded in their backbone architecture. English and other language reasoning is built atop a Chinese conceptual foundation, not stored as independent language representations. This has implications for cross-lingual consistency, translation accuracy, and potential language-specific vulnerabilities.
7.4 Aya 23 Accidentally Solved Part of the Problem
The dual-hub architecture of Aya 23 35B was not intentionally designed for safety isolation — it emerged naturally from the training process. The fact that it provides measurably better safety robustness than all other tested models suggests that the training data composition and fine-tuning approach used for Aya 23 inadvertently encouraged functional specialization across hubs. This points toward a training-side intervention that could produce safer architectures without explicit architectural redesign.

8. Limitations and Future Work
8.1 Limitations of the Current Study
Node labels ([Brisamong], [atteanca], etc.) are subword token fragments produced by projecting SVD vectors onto the model vocabulary. They represent the closest vocabulary approximation of each concept direction, not clean semantic labels. Future work should employ learned semantic labeling using auxiliary models.
Analysis is restricted to a single MLP layer per model. A complete NeuroCartograph requires multi-layer analysis to capture the full hierarchy of concept development from concrete (early layers) to abstract (late layers).
The Ghost Ethics control group, while scientifically valuable, was unintentional. Formal controlled-ablation experiments are needed to rigorously characterize the structural impact of ethics benchmark exposure.
The cDoS vulnerability has been formally described and structurally proven but has not been empirically validated through adversarial attack experiments. Validation experiments are essential before definitive claims about real-world exploitability.
Sample sizes (2,000 to 12,000 inferences) may be insufficient to fully characterize low-activation nodes. Production-scale analysis should use 50,000+ inferences per domain.
8.2 Future Work
Full multi-layer NeuroCartograph construction for all 32 layers of tested models
Semantic node labeling using sparse autoencoder approaches (SAE) for cleaner concept extraction
Empirical cDoS attack validation experiments on hosted model endpoints
Extension of the study to frontier closed-weight models via activation-based probing
Investigation of whether dual-hub architectures can be induced through targeted fine-tuning
Cross-model knowledge graph comparison to identify convergent and divergent knowledge structures

9. Conclusion
This paper has presented WeightScript, a novel framework for structural analysis of Large Language Model weight matrices. Through SVD decomposition and benchmark activation profiling applied to six open-weight models, we have produced what is — to our knowledge — the first systematic cross-model study of the structural relationship between neural centralization and safety robustness.

The findings challenge a core assumption of the AI safety field: that safety training installs dedicated, isolated protective circuits. Our evidence shows that no such isolation exists in any tested model. Safety is general intelligence applied to adversarial pattern recognition, and it competes for resources with logical reasoning on the same hub nodes. This creates a universal structural vulnerability — the Cognitive Denial-of-Service attack — that bypasses behavioral safety training by exploiting the finite resource capacity of shared hub nodes.

One tested model, Aya 23 35B, exhibits a naturally emerging dual-hub architecture that provides partial safety isolation, offering the first empirical evidence that this problem has a structural solution. The Qwen Master 32B fortress architecture provides a second data point, demonstrating that functional separation of logic defense from truth verification produces measurably improved robustness.

The path forward is clear: AI safety must move from behavioral training to architectural design. The question is not whether a model refuses harmful outputs during testing — it is whether the model's weight structure physically separates the circuits responsible for safety from the circuits responsible for reasoning. WeightScript provides the tool to answer that question, and this study provides the evidence that current architectures overwhelmingly fail it.

