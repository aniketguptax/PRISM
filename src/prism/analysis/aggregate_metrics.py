import pandas as pd

df = pd.read_csv("results/branching_vs_k/metrics.csv")

# Extract k from "last_k"
df["k"] = df["representation"].str.replace("last_", "").astype(int)

summary = (
    df.groupby("k")
      .agg(
          logloss_mean=("logloss", "mean"),
          logloss_std=("logloss", "std"),
          n_states_mean=("n_states", "mean"),
          n_states_std=("n_states", "std"),
          Cmu_mean=("C_mu", "mean"),
          Cmu_std=("C_mu", "std"),
          unif_mean=("unifilarity_score", "mean"),
          unif_std=("unifilarity_score", "std"),
          branch_mean=("branch_entropy", "mean"),
          branch_std=("branch_entropy", "std"),
      )
      .reset_index()
)

print(summary)