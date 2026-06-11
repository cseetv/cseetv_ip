import csv, matplotlib.pyplot as plt, numpy as np, os
OUT = "experiments/results"

# ── 그림 5: Ablation F1 막대 ──
names = ["Baseline", "+CLAHE", "+Gauss\n+Median", "+Morph\n★", "+Shadow\nRemoval"]
f1s = [0.407, 0.407, 0.403, 0.471, 0.137]
colors = ["#94A3B8","#94A3B8","#94A3B8","#4F46E5","#EF4444"]
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(range(len(names)), f1s, color=colors, width=0.6)
for b,v in zip(bars,f1s):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Pixel F1-Score", fontsize=12); ax.set_ylim(0, 0.55)
ax.set_title("Ablation Study: F1 by Pipeline Stage", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig5_ablation_f1.png", dpi=200)
print("→ fig5_ablation_f1.png")

# ── 그림 6: T vs P/R/F1 ──
Ts = [10,15,20,25,30,35,40,50]
prec = [0.360,0.386,0.406,0.424,0.440,0.455,0.466,0.480]
rec = [0.663,0.613,0.571,0.531,0.493,0.455,0.415,0.338]
f1 = [0.467,0.474,0.475,0.471,0.465,0.455,0.439,0.397]
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(Ts, prec, "b-o", label="Precision", linewidth=2)
ax.plot(Ts, rec, "r-s", label="Recall", linewidth=2)
ax.plot(Ts, f1, "g-^", label="F1-Score", linewidth=2)
ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="Optimal T=20")
ax.set_xlabel("Threshold (T)", fontsize=12); ax.set_ylabel("Score", fontsize=12)
ax.set_title("Threshold vs Performance (Shadow OFF)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig6_threshold.png", dpi=200)
print("→ fig6_threshold.png")

# ── 그림 7: PR Curve ──
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(rec, prec, "b-o", linewidth=2, markersize=7)
for t,r,p in zip(Ts,rec,prec):
    ax.annotate(f"T={t}", (r,p), textcoords="offset points", xytext=(6,6), fontsize=9)
ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.set_xlim(0.25,0.75); ax.set_ylim(0.3,0.55); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig7_pr_curve.png", dpi=200)
print("→ fig7_pr_curve.png")

# ── 그림 8: 기법별 F1 비교 ──
names2 = ["Frame\nDiff", "Running\nAvg ★", "3-Frame\nDiff", "Canny\nEdge", "MOG2\n(AI)"]
f1s2 = [0.470, 0.566, 0.178, 0.000, 0.614]
colors2 = ["#94A3B8","#4F46E5","#EF4444","#EF4444","#F59E0B"]
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(range(len(names2)), f1s2, color=colors2, width=0.6)
for b,v in zip(bars,f1s2):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(names2))); ax.set_xticklabels(names2, fontsize=10)
ax.set_ylabel("Pixel F1-Score", fontsize=12); ax.set_ylim(0, 0.72)
ax.set_title("Advanced Techniques + AI Comparison", fontsize=14, fontweight="bold")
ax.axhline(y=0.566, color="#4F46E5", linestyle="--", alpha=0.3, label="Best Traditional")
ax.axhline(y=0.614, color="#F59E0B", linestyle="--", alpha=0.3, label="Best AI (MOG2)")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig8_advanced_f1.png", dpi=200)
print("→ fig8_advanced_f1.png")

print("\n✅ 그래프 4장 생성 완료!")