#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
#%%
df = pd.read_csv("data/MCA.csv")
#%%
x = df["0"]
y = df["1"]
colors = df["class"]

# Create a dictionary to map class values to colors
color_map = {0: "blue", 1: "orange"}

# Create a list of colors based on the 'class' column
scatter_colors = [color_map[c] for c in colors]

# # Create the scatter plot
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(x, y, c=scatter_colors)

# # Add a legend
# legend_labels = {0: "Benign", 1: "Malware"}
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=8) for i in color_map]
# plt.legend(handles, [legend_labels[i] for i in color_map], title="Class")

# # Add labels and title
# plt.xlabel("1st Principal Component")
# plt.ylabel("2nd Principal Component")
# plt.title("2d MCA Data")

# # Center the origin at (0, 0)
# max_range = max(max(abs(x)), max(abs(y)))  # Find the maximum absolute value
# plt.xlim([-max_range, max_range])
# plt.ylim([-max_range, max_range])

# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)

# # Save the plot
# plt.savefig("outs/mca_scatter.png")

# # plt.show()
#%%
# fig = px.scatter(df, x="0", y="1", color="class", color_discrete_map=color_map, title="2d MCA Data", labels={"0": "1st Principal Component", "1": "2nd Principal Component"}, width=800, height=600)
# fig.write_html("outs/mca_scatter.html")

#%%

df['class'] = df['class'].map({0: 'Benign', 1: 'Malware'})

vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='black').encode(x='x:Q')
hline = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black').encode(y='y:Q')

scatter = alt.Chart(df).mark_circle().encode(
    x=alt.X('0', title='1st Principal component'),
    y=alt.Y('1', title='2nd Principal component'),
    color=alt.Color('class:N', scale=alt.Scale(range=['blue', 'orange']))
)
chart = alt.layer(scatter, vline, hline).properties(width=1200, height=800).interactive()

chart.save("outs/mca_scatter.html")