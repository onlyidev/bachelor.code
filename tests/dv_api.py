#%%
import yaml
export = dict(
    malgan = dict(
        id = 'asldkj'
    )
)

with open("malgan.yaml", "w") as f:
    yaml.dump(export, f)