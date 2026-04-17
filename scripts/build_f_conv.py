import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load raw data — no processing
df = pd.read_csv("/Users/sena/grad_school/Research/ALP/tracking-ALPs/data/total_x_X0.txt", header=None, names=["eta", "t_over_X0"])
df = df.sort_values("eta").reset_index(drop=True)

# Bethe-Heitler on raw points
df["f_conv_true"]  = 1.0 - np.exp(-(7.0/9.0) * df["t_over_X0"])
df["f_conv_total"] = df["f_conv_true"] * 0.70 + (1.0 - df["f_conv_true"]) * 0.05

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax1.plot(df["eta"], df["t_over_X0"], "k.", ms=3)
ax1.set_ylabel("x/X₀")
ax1.set_title("CMS Phase-1 material budget — raw digitized")
ax1.grid(True, alpha=0.3)

ax2.plot(df["eta"], df["f_conv_true"],  "b.", ms=3, label=r"$f^\mathrm{true}_\mathrm{conv}$")
ax2.plot(df["eta"], df["f_conv_total"], "r.", ms=3, label=r"$f^\mathrm{total}_\mathrm{conv}$")
ax2.set_xlabel("η")
ax2.set_ylabel("Conversion probability")
ax2.set_title("Photon conversion probability — CMS Phase-1")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cms_fconv_vs_eta.pdf", dpi=150)
plt.savefig("cms_fconv_vs_eta.png", dpi=150)
print("Done.")
print(df[["eta","t_over_X0","f_conv_true","f_conv_total"]].to_string())