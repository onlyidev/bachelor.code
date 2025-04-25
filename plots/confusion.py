# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

#%%
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
def plot_confusion_matrix(
    confusion_matrix, class_names=None, title="Klasifikacijos matrica", unbalanced=False
):
    """
    Plot a confusion matrix using matplotlib and seaborn.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix to plot
        class_names (list, optional): List of class names for axis labels
        title (str, optional): Title for the plot
    """
    if unbalanced:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:,None]
    
    # Create figure and axes
    plt.figure(figsize=(8, 6))

    # If no class names provided, use numerical indices
    if class_names is None:
        class_names = [str(i) for i in range(len(confusion_matrix))]

    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,  # Show numbers in cells
        fmt="d" if not unbalanced else ".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(title)
    plt.ylabel("Tikra klasė")
    plt.xlabel("Prognozuojama klasė")
    plt.tight_layout()


# %% 2x2 normal case
data = json.load(open("../metrics/normal.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška"]

plot_confusion_matrix(confusion_matrix, classes, unbalanced=True)
plt.savefig("./normal_2x2.png", dpi=300)
#plt.show()

# %% 2x2 LIME
data = json.load(open("../metrics/lime_cat.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška"]

plot_confusion_matrix(confusion_matrix, classes, unbalanced=True)
plt.savefig("./lime_2x2.png", dpi=300)
#plt.show()

# %% 2x2 synthesis
data = json.load(open("../metrics/lime.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška"]

plot_confusion_matrix(confusion_matrix, classes, unbalanced=True)
plt.savefig("./synthesis_2x2.png", dpi=300)
#plt.show()
# %% 2x2 MCA
data = json.load(open("../metrics/mca_equiv.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška"]

plot_confusion_matrix(confusion_matrix, classes, unbalanced=True)
plt.savefig("./mca_2x2.png", dpi=300)
#plt.show()

# %% 3x3 LIME
data = json.load(open("../metrics/lime_cat_obf.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška", "Obfuskuota"]

plot_confusion_matrix(confusion_matrix, classes)
plt.savefig("./lime_3x3.png", dpi=300)
#plt.show()

# %% 3x3 synthesis
data = json.load(open("../metrics/lime_obf.json", "r", encoding="utf-8"))
confusion_matrix = np.array(data["confusion_matrix"])

classes = ["Nekenkėjiška", "Kenkėjiška", "Obfuskuota"]

plot_confusion_matrix(confusion_matrix, classes)
plt.savefig("./synthesis_3x3.png", dpi=300)
# plt.show()
