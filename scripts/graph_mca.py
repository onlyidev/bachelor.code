#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("data/MCA.csv")
df.head()
#%%
x = df["Low API Density"]
y = df["High API Density"]
colors = df["class"]

# Create a dictionary to map class values to colors
color_map = {0: "blue", 1: "orange"}

# Create a list of colors based on the 'class' column
scatter_colors = [color_map[c] for c in colors]

# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=scatter_colors)

# Add a legend
legend_labels = {0: "Benign", 1: "Malware"}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=8) for i in color_map]
plt.legend(handles, [legend_labels[i] for i in color_map], title="Class")

# Add labels and title
plt.xlabel("Low API Density")
plt.ylabel("High API Density")
plt.title("Scatter Plot of API Densities")

# Center the origin at (0, 0)
max_range = max(max(abs(x)), max(abs(y)))  # Find the maximum absolute value
plt.xlim([-max_range, max_range])
plt.ylim([-max_range, max_range])

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Save the plot
plt.savefig("outs/mca_scatter.png")

# plt.show()
