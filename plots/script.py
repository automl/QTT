import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


title = "FGVC-Aircraft", "Stanford-Cars", "Oxford-Flowers-102", "Imagenette"
marker = ["o", "s", "D", "v"]
every_step = 16, 16, 8, 8
label = "QT", "QT (step 2)", "Random", "Random (step 2)"
colour = "blue", "orange", "green", "red"

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for i, dir in enumerate(title):
    ax = axes.flatten()[i]
    for j, sub_dir in enumerate(label):
        file_path = os.path.join("data", dir, sub_dir) + ".csv"
        df = pd.read_csv(file_path, index_col=False)
        df = pd.melt(df, id_vars=["time"], var_name="seed", value_name="score")
        n_samples = len(df)
        step = every_step[i]
        g = sns.lineplot(
                df,
                x="time",
                y="score",
                label=sub_dir,
                marker=marker[j],
                markerfacecolor="white",
                markeredgecolor=colour[j],
                markeredgewidth=1.5,
                markevery=slice(10 * j, n_samples, step),
                ax=ax,
                color=colour[j],
                errorbar="ci",
            )
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title[j])
    ax.legend([], [], frameon=False)

plt.tight_layout()
plt.subplots_adjust(
    left=0.08, hspace=0.27, bottom=0.18
)  # Add vertical space between plots and adjust bottom space
# Set common labels
fig.text(0.5, 0.1, "Time (s)", ha="center", va="center", fontsize=12)
fig.text(
    0.04,
    0.5,
    "Evaluation Accuracy (%)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=12,
)
# add a common legend
handles = axes.flatten()[0].get_legend_handles_labels()[0]
fig.legend(handles, label, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
plt.savefig("plot.pdf", format="pdf")