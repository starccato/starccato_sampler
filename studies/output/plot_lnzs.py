import matplotlib.pyplot as plt
import numpy as np

LNZ_FILE = "lnz_results.txt"
# headers NS_LnZ SS_LnZ

lnz = np.loadtxt(LNZ_FILE, delimiter=" ", skiprows=1)
ns_lnzs = lnz[:, 0]
ss_lnzs = lnz[:, 1]

# plot two violin plots
fig, ax = plt.subplots()
ax.violinplot([ns_lnzs, ss_lnzs], showmeans=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Nested Sampling", "Stepping Stone"])
plt.ylabel("LnZ")
plt.savefig("lnz_violin.png", dpi=300, bbox_inches="tight")
