import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUNC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order: run 0..3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 1.2, "legend.frameon": False})
