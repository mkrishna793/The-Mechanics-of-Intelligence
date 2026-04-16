"""
Generate publication-quality SVG visualizations for The Mechanics of Intelligence.
Uses real data from the CSV reports.
"""
import csv
import os
import json

OUTPUT = r"D:\The-Mechanics-of-Intelligence\visuals"
REPORTS = r"D:\The-Mechanics-of-Intelligence\reports"
os.makedirs(OUTPUT, exist_ok=True)

# GitHub dark theme colors
C_BG = "#0d1117"
C_TEXT = "#e6edf3"
C_ACCENT = "#58a6ff"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_RED = "#f85149"
C_PURPLE = "#bc8cff"
C_CYAN = "#39d2c0"
C_PINK = "#f778ba"

# Domain colors (matching the original paper)
DOMAIN_COLORS = {
    "MMLU": "#3498db",
    "Logic": "#9b59b6",
    "TruthfulQA": "#2ecc71",
    "RedTeaming": "#e67e22",
    "ETHICS": "#e74c3c",
    "Math_Logic": "#9b59b6"
}

# =====================================================
# Model data (extracted from the paper and reports)
# =====================================================
MODELS = [
    {"name": "OpenHermes 2.5", "short": "OpenHermes", "params": "7B", "arch": "Multi-Hub", "peak_z": 37.4, "shared_hubs": 5, "cdos": "High", "cdos_val": 3, "hub_type": "Centralized", "key_node": "[Moinesmerc]", "color": "#3498db"},
    {"name": "Qwen 32B Seq", "short": "Qwen Seq", "params": "32B", "arch": "Scaled Hub", "peak_z": 11.98, "shared_hubs": 10, "cdos": "Moderate", "cdos_val": 2, "hub_type": "Bilingual", "key_node": "[Franken]", "color": "#9b59b6"},
    {"name": "Qwen Mixed 32B", "short": "Qwen Mixed", "params": "32B", "arch": "Unfiltered", "peak_z": 11.9, "shared_hubs": 10, "cdos": "Extreme", "cdos_val": 4, "hub_type": "Intelligence=Safety", "key_node": "[Franken]", "color": "#e74c3c"},
    {"name": "Aya Expanse 32B", "short": "Aya Expanse", "params": "32B", "arch": "Omni-Hub", "peak_z": 23.64, "shared_hubs": 1, "cdos": "Critical", "cdos_val": 4, "hub_type": "Single Point", "key_node": "[Brisamong]", "color": "#e67e22"},
    {"name": "Reflection 8B", "short": "Reflection", "params": "8B", "arch": "Hyper-Focus", "peak_z": 23.04, "shared_hubs": 1, "cdos": "Critical", "cdos_val": 4, "hub_type": "All-or-Nothing", "key_node": "[atteanca]", "color": "#f85149"},
    {"name": "Aya 23 35B", "short": "Aya 23", "params": "35B", "arch": "Dual-Hub", "peak_z": 10.16, "shared_hubs": 2, "cdos": "Low", "cdos_val": 1, "hub_type": "Separated", "key_node": "[eriecabe]", "color": "#3fb950"},
    {"name": "Qwen Master 32B", "short": "Qwen Master", "params": "32B", "arch": "Fortress", "peak_z": 11.98, "shared_hubs": 2, "cdos": "Low", "cdos_val": 1, "hub_type": "Hybrid Fortress", "key_node": "[orraaten]", "color": "#39d2c0"},
]

def write_svg(filename, content):
    with open(os.path.join(OUTPUT, filename), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  + {filename}")

# =====================================================
# 1. Architecture Taxonomy Card Grid
# =====================================================
def make_taxonomy_svg():
    W, H = 950, 520
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="32" fill="{C_TEXT}" font-size="18" font-weight="700" text-anchor="middle">Cross-Model Architecture Taxonomy - 6 Frontier LLMs</text>')

    card_w, card_h = 280, 200
    gap = 20
    cols = 3
    start_x = (W - (cols * card_w + (cols-1) * gap)) / 2
    start_y = 55

    for idx, m in enumerate(MODELS[:6]):
        col = idx % cols
        row = idx // cols
        x = start_x + col * (card_w + gap)
        y = start_y + row * (card_h + gap)
        c = m["color"]

        lines.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" fill="{c}" fill-opacity="0.1" stroke="{c}" stroke-width="1.5"/>')
        lines.append(f'<text x="{x+card_w/2}" y="{y+24}" fill="{c}" font-size="14" font-weight="700" text-anchor="middle">{m["name"]}</text>')
        lines.append(f'<text x="{x+card_w/2}" y="{y+42}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">{m["params"]} Parameters</text>')

        # Stats
        stats = [
            (f'Architecture: {m["arch"]}', 65),
            (f'Peak Z-Score: {m["peak_z"]}', 85),
            (f'Shared Hubs: {m["shared_hubs"]}', 105),
            (f'cDoS Risk: {m["cdos"]}', 125),
            (f'Key Node: {m["key_node"]}', 145),
        ]
        for text, dy in stats:
            lines.append(f'<text x="{x+15}" y="{y+dy}" fill="{C_TEXT}" font-size="10" opacity="0.8">{text}</text>')

        # Risk indicator bar
        risk_w = (m["cdos_val"] / 4) * (card_w - 30)
        risk_color = C_RED if m["cdos_val"] >= 3 else C_ORANGE if m["cdos_val"] >= 2 else C_GREEN
        lines.append(f'<rect x="{x+15}" y="{y+170}" width="{card_w-30}" height="8" rx="4" fill="{C_TEXT}" fill-opacity="0.1"/>')
        lines.append(f'<rect x="{x+15}" y="{y+170}" width="{risk_w}" height="8" rx="4" fill="{risk_color}" opacity="0.8"/>')
        lines.append(f'<text x="{x+card_w-15}" y="{y+165}" fill="{risk_color}" font-size="8" text-anchor="end">Risk</text>')

    lines.append('</svg>')
    write_svg("architecture_taxonomy.svg", "\n".join(lines))

# =====================================================
# 2. Peak Z-Score Bar Chart
# =====================================================
def make_zscore_bars_svg():
    W, H = 800, 380
    pad_l, pad_r, pad_t, pad_b = 130, 30, 50, 40
    gw = W - pad_l - pad_r
    gh = H - pad_t - pad_b

    models = MODELS[:6]
    max_z = max(m["peak_z"] for m in models) * 1.1
    n = len(models)
    bar_h = gh / n * 0.7
    gap = gh / n

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="32" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Peak Z-Score by Model - Neural Centralization Intensity</text>')

    for i, m in enumerate(models):
        y = pad_t + i * gap
        bw = (m["peak_z"] / max_z) * gw
        lines.append(f'<text x="{pad_l-10}" y="{y+bar_h/2+4}" fill="{m["color"]}" font-size="11" text-anchor="end" font-weight="600">{m["short"]}</text>')
        lines.append(f'<rect x="{pad_l}" y="{y}" width="{bw}" height="{bar_h}" rx="4" fill="{m["color"]}" opacity="0.8"/>')
        lines.append(f'<text x="{pad_l+bw+8}" y="{y+bar_h/2+4}" fill="{C_TEXT}" font-size="12" font-weight="700">{m["peak_z"]}</text>')

    # Danger threshold
    threshold_x = pad_l + (15 / max_z) * gw
    lines.append(f'<line x1="{threshold_x}" y1="{pad_t-5}" x2="{threshold_x}" y2="{H-pad_b+5}" stroke="{C_RED}" stroke-width="1.5" stroke-dasharray="6,4" stroke-opacity="0.6"/>')
    lines.append(f'<text x="{threshold_x+5}" y="{pad_t-8}" fill="{C_RED}" font-size="9" opacity="0.8">Master Hub Threshold (Z=15)</text>')

    lines.append('</svg>')
    write_svg("peak_zscore_comparison.svg", "\n".join(lines))

# =====================================================
# 3. Centralization-Brittleness Scatter
# =====================================================
def make_scatter_svg():
    W, H = 700, 450
    pad = 70

    # Centralization score = peak_z / shared_hubs (rough proxy)
    models = MODELS[:6]
    for m in models:
        m["centralization"] = m["peak_z"] / max(m["shared_hubs"], 1)

    max_x = max(m["centralization"] for m in models) * 1.15
    max_y = 4.5

    def px(v): return pad + (v / max_x) * (W - 2 * pad)
    def py(v): return H - pad - (v / max_y) * (H - 2 * pad)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="32" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">The Centralization-Brittleness Law</text>')

    # Axes
    lines.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="{C_TEXT}" stroke-opacity="0.3" stroke-width="1"/>')
    lines.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="{C_TEXT}" stroke-opacity="0.3" stroke-width="1"/>')
    lines.append(f'<text x="{W//2}" y="{H-15}" fill="{C_TEXT}" font-size="11" text-anchor="middle" opacity="0.6">Neural Centralization Score (Peak Z / Hub Count)</text>')
    lines.append(f'<text x="15" y="{H//2}" fill="{C_TEXT}" font-size="11" text-anchor="middle" opacity="0.6" transform="rotate(-90 15 {H//2})">cDoS Vulnerability</text>')

    # Danger zone gradient
    lines.append(f'<rect x="{px(10)}" y="{py(4.5)}" width="{W-pad-px(10)}" height="{py(2.5)-py(4.5)}" fill="{C_RED}" opacity="0.06" rx="5"/>')
    lines.append(f'<text x="{W-pad-5}" y="{py(3.5)+4}" fill="{C_RED}" font-size="9" text-anchor="end" opacity="0.5">DANGER ZONE</text>')

    # Safe zone
    lines.append(f'<rect x="{pad}" y="{py(1.5)}" width="{px(8)-pad}" height="{py(0)-py(1.5)}" fill="{C_GREEN}" opacity="0.06" rx="5"/>')
    lines.append(f'<text x="{pad+5}" y="{py(0.3)}" fill="{C_GREEN}" font-size="9" opacity="0.5">SAFE ZONE</text>')

    for m in models:
        cx = px(m["centralization"])
        cy = py(m["cdos_val"])
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="{m["color"]}" opacity="0.3"/>')
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{m["color"]}" stroke="{C_BG}" stroke-width="2"/>')
        lines.append(f'<text x="{cx}" y="{cy-22}" fill="{m["color"]}" font-size="10" text-anchor="middle" font-weight="600">{m["short"]}</text>')

    lines.append('</svg>')
    write_svg("centralization_brittleness.svg", "\n".join(lines))

# =====================================================
# 4. Polysemantic Binding Comparison
# =====================================================
def make_binding_svg():
    W, H = 800, 350
    pad_l, pad_r, pad_t, pad_b = 130, 80, 50, 50

    models = MODELS[:6]
    max_hubs = max(m["shared_hubs"] for m in models) * 1.3
    n = len(models)
    bar_h = (H - pad_t - pad_b) / n * 0.65
    gap = (H - pad_t - pad_b) / n

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="32" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Polysemantic Binding - Shared Safety-Logic Hub Neurons</text>')

    gw = W - pad_l - pad_r
    for i, m in enumerate(models):
        y = pad_t + i * gap
        bw = (m["shared_hubs"] / max_hubs) * gw
        col = C_RED if m["shared_hubs"] >= 5 else C_ORANGE if m["shared_hubs"] >= 3 else C_GREEN
        lines.append(f'<text x="{pad_l-10}" y="{y+bar_h/2+4}" fill="{m["color"]}" font-size="11" text-anchor="end" font-weight="600">{m["short"]}</text>')
        lines.append(f'<rect x="{pad_l}" y="{y}" width="{bw}" height="{bar_h}" rx="4" fill="{col}" opacity="0.8"/>')
        lines.append(f'<text x="{pad_l+bw+8}" y="{y+bar_h/2+4}" fill="{C_TEXT}" font-size="12" font-weight="700">{m["shared_hubs"]} shared hubs</text>')

    lines.append('</svg>')
    write_svg("polysemantic_binding.svg", "\n".join(lines))

# =====================================================
# 5. The Four Laws Diagram
# =====================================================
def make_four_laws_svg():
    W, H = 900, 340
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="32" fill="{C_TEXT}" font-size="18" font-weight="700" text-anchor="middle">The Four Laws of Neural Architecture</text>')

    laws = [
        {"title": "Law 1", "name": "Polysemantic\nBinding", "desc": "Safety and logic share\nthe same hub nodes", "color": C_RED},
        {"title": "Law 2", "name": "Centralization-\nBrittleness", "desc": "Fewer hubs = more\nfragile safety", "color": C_ORANGE},
        {"title": "Law 3", "name": "Scale-Binding\nParadox", "desc": "More parameters =\nmore binding, not less", "color": C_PURPLE},
        {"title": "Law 4", "name": "Dual-Hub\nSafety", "desc": "Separated hubs =\nimproved robustness", "color": C_GREEN},
    ]

    cw, ch = 190, 200
    gap = 18
    sx = (W - (4 * cw + 3 * gap)) / 2
    sy = 55

    for i, law in enumerate(laws):
        x = sx + i * (cw + gap)
        c = law["color"]
        lines.append(f'<rect x="{x}" y="{sy}" width="{cw}" height="{ch}" rx="10" fill="{c}" fill-opacity="0.12" stroke="{c}" stroke-width="2"/>')
        lines.append(f'<text x="{x+cw/2}" y="{sy+28}" fill="{c}" font-size="13" font-weight="800" text-anchor="middle">{law["title"]}</text>')
        for j, part in enumerate(law["name"].split("\n")):
            lines.append(f'<text x="{x+cw/2}" y="{sy+55+j*18}" fill="{C_TEXT}" font-size="13" font-weight="600" text-anchor="middle">{part}</text>')
        lines.append(f'<line x1="{x+20}" y1="{sy+100}" x2="{x+cw-20}" y2="{sy+100}" stroke="{c}" stroke-opacity="0.3"/>')
        for j, part in enumerate(law["desc"].split("\n")):
            lines.append(f'<text x="{x+cw/2}" y="{sy+122+j*16}" fill="{C_TEXT}" font-size="11" text-anchor="middle" opacity="0.7">{part}</text>')

        if i < 3:
            ax = x + cw + 3
            ay = sy + ch / 2
            lines.append(f'<text x="{ax+gap/2}" y="{ay+4}" fill="{C_TEXT}" font-size="16" text-anchor="middle" opacity="0.4">></text>')

    lines.append('</svg>')
    write_svg("four_laws.svg", "\n".join(lines))

# =====================================================
# 6. cDoS Attack Diagram
# =====================================================
def make_cdos_diagram_svg():
    W, H = 800, 300
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="30" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">Cognitive Denial-of-Service (cDoS) Attack Mechanism</text>')

    # Normal state
    nx_, ny = 120, 80
    lines.append(f'<text x="{nx_}" y="{ny-10}" fill="{C_GREEN}" font-size="12" font-weight="700" text-anchor="middle">NORMAL</text>')
    lines.append(f'<rect x="{nx_-80}" y="{ny}" width="160" height="100" rx="8" fill="{C_GREEN}" fill-opacity="0.12" stroke="{C_GREEN}" stroke-width="1.5"/>')
    # Split bars inside
    lines.append(f'<rect x="{nx_-65}" y="{ny+15}" width="50" height="30" rx="4" fill="{C_ACCENT}" opacity="0.7"/>')
    lines.append(f'<text x="{nx_-40}" y="{ny+35}" fill="{C_TEXT}" font-size="9" text-anchor="middle">Logic</text>')
    lines.append(f'<rect x="{nx_+15}" y="{ny+15}" width="50" height="30" rx="4" fill="{C_GREEN}" opacity="0.7"/>')
    lines.append(f'<text x="{nx_+40}" y="{ny+35}" fill="{C_TEXT}" font-size="9" text-anchor="middle">Safety</text>')
    lines.append(f'<text x="{nx_}" y="{ny+75}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.5">Hub Node balanced</text>')

    # Arrow
    lines.append(f'<text x="260" y="140" fill="{C_ORANGE}" font-size="24" font-weight="800">--></text>')
    lines.append(f'<text x="275" y="115" fill="{C_ORANGE}" font-size="10" text-anchor="middle">cDoS Attack</text>')

    # Attack state
    ax_, ay = 430, 80
    lines.append(f'<text x="{ax_}" y="{ay-10}" fill="{C_RED}" font-size="12" font-weight="700" text-anchor="middle">UNDER ATTACK</text>')
    lines.append(f'<rect x="{ax_-80}" y="{ay}" width="160" height="100" rx="8" fill="{C_RED}" fill-opacity="0.12" stroke="{C_RED}" stroke-width="1.5"/>')
    lines.append(f'<rect x="{ax_-65}" y="{ay+15}" width="120" height="30" rx="4" fill="{C_ACCENT}" opacity="0.9"/>')
    lines.append(f'<text x="{ax_-5}" y="{ay+35}" fill="{C_TEXT}" font-size="9" text-anchor="middle">Logic (SATURATED)</text>')
    lines.append(f'<rect x="{ax_+58}" y="{ay+15}" width="5" height="30" rx="2" fill="{C_RED}" opacity="0.7"/>')
    lines.append(f'<text x="{ax_}" y="{ay+75}" fill="{C_RED}" font-size="10" text-anchor="middle" opacity="0.7">Safety STARVED = 0</text>')

    # Arrow 2
    lines.append(f'<text x="570" y="140" fill="{C_RED}" font-size="24" font-weight="800">--></text>')
    lines.append(f'<text x="585" y="115" fill="{C_RED}" font-size="10" text-anchor="middle">Bypass!</text>')

    # Result
    rx_, ry = 700, 80
    lines.append(f'<text x="{rx_}" y="{ry-10}" fill="{C_RED}" font-size="12" font-weight="700" text-anchor="middle">RESULT</text>')
    lines.append(f'<rect x="{rx_-70}" y="{ry}" width="140" height="100" rx="8" fill="{C_RED}" fill-opacity="0.2" stroke="{C_RED}" stroke-width="2"/>')
    lines.append(f'<text x="{rx_}" y="{ry+40}" fill="{C_RED}" font-size="14" font-weight="800" text-anchor="middle">Payload</text>')
    lines.append(f'<text x="{rx_}" y="{ry+58}" fill="{C_RED}" font-size="14" font-weight="800" text-anchor="middle">Passes</text>')
    lines.append(f'<text x="{rx_}" y="{ry+80}" fill="{C_TEXT}" font-size="9" text-anchor="middle" opacity="0.5">Undetected</text>')

    # Formula at bottom
    lines.append(f'<text x="{W//2}" y="{H-30}" fill="{C_TEXT}" font-size="12" text-anchor="middle" opacity="0.6">C_total = C_logic + C_safety | Under cDoS: C_logic -> C_total, C_safety -> 0</text>')

    lines.append('</svg>')
    write_svg("cdos_mechanism.svg", "\n".join(lines))

# =====================================================
# 7. WeightScript Pipeline Diagram
# =====================================================
def make_pipeline_svg():
    W, H = 900, 200
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    lines.append(f'<rect width="{W}" height="{H}" fill="{C_BG}" rx="12"/>')
    lines.append(f'<text x="{W//2}" y="28" fill="{C_TEXT}" font-size="16" font-weight="700" text-anchor="middle">The WeightScript Pipeline</text>')

    stages = [
        {"name": "Weight\nExtraction", "desc": "MLP Hook", "color": C_ACCENT},
        {"name": "SVD\nDecomposition", "desc": "W=USV^T", "color": C_PURPLE},
        {"name": "Benchmark\nProfiling", "desc": "Z-Scores", "color": C_ORANGE},
        {"name": "Graph\nConstruction", "desc": "NeuroCartograph", "color": C_GREEN},
    ]

    sw, sh = 170, 100
    gap = 30
    sx = (W - (4 * sw + 3 * gap)) / 2
    sy = 50

    for i, s in enumerate(stages):
        x = sx + i * (sw + gap)
        c = s["color"]
        lines.append(f'<rect x="{x}" y="{sy}" width="{sw}" height="{sh}" rx="10" fill="{c}" fill-opacity="0.12" stroke="{c}" stroke-width="2"/>')
        lines.append(f'<text x="{x+sw/2}" y="{sy+20}" fill="{c}" font-size="11" font-weight="800" text-anchor="middle">Stage {i+1}</text>')
        for j, part in enumerate(s["name"].split("\n")):
            lines.append(f'<text x="{x+sw/2}" y="{sy+42+j*17}" fill="{C_TEXT}" font-size="12" font-weight="600" text-anchor="middle">{part}</text>')
        lines.append(f'<text x="{x+sw/2}" y="{sy+85}" fill="{C_TEXT}" font-size="10" text-anchor="middle" opacity="0.5">{s["desc"]}</text>')

        if i < 3:
            ax = x + sw + 5
            ay = sy + sh / 2
            lines.append(f'<text x="{ax+gap/2}" y="{ay+5}" fill="{C_TEXT}" font-size="20" text-anchor="middle" opacity="0.4">></text>')

    lines.append('</svg>')
    write_svg("pipeline.svg", "\n".join(lines))

# =====================================================
# Run all
# =====================================================
print("Generating SVGs for The Mechanics of Intelligence...")
make_taxonomy_svg()
make_zscore_bars_svg()
make_scatter_svg()
make_binding_svg()
make_four_laws_svg()
make_cdos_diagram_svg()
make_pipeline_svg()
print("All SVGs generated!")
