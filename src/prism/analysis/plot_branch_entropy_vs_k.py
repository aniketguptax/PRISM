import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "results/branching_vs_k/metrics.csv"
OUT_PDF = "figures/branch_entropy_vs_k.pdf"
OUT_PNG = "figures/branch_entropy_vs_k.png"

df = pd.read_csv(IN_CSV)

# representation is like "last_3"
df["k"] = df["representation"].str.replace("last_", "", regex=False).astype(int)

summary = (
    df.groupby("k")["branch_entropy"]
      .agg(["mean", "std"])
      .reset_index()
      .sort_values("k")
)

plt.figure(figsize=(4.8, 3.2))
plt.errorbar(
    summary["k"],
    summary["mean"],
    yerr=summary["std"],
    fmt="o-",
    capsize=3,
)

plt.annotate(
    "Onset of non-unifilarity",
    xy=(3, 0.195),
    xytext=(3.3, 0.08),
    arrowprops=dict(arrowstyle="->", alpha=0.6),
    fontsize=9,
)

plt.xlabel("Representation length $k$")
plt.ylabel("Mean branching entropy (bits)")
plt.tight_layout()

# Save both (PDF is best for LaTeX)
plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Wrote {OUT_PDF} and {OUT_PNG}")