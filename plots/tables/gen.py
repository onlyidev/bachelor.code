# %% imports
import json
import pandas as pd

# %% Directory change
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %%
def report_to_record(report):
    return [
        report["precision"],
        report["recall"],
        report["f1-score"],
    ]


# %%
def report_to_df_2d(data):
    rep = data["classification_report"]
    lst = [
        ["Nekenkėjiška", *report_to_record(rep["0"])],
        ["Kenkėjiška", *report_to_record(rep["1"])],
        ["Svertinis vidurkis", *report_to_record(rep["weighted avg"])],
    ]

    return pd.DataFrame(
        lst, columns=("Klasė", "Preciziškumas", "Atkūrimas", "F1")
    )
    
def report_to_df_3d(data):
    rep = data["classification_report"]
    lst = [
        ["Nekenkėjiška", *report_to_record(rep["0"])],
        ["Kenkėjiška", *report_to_record(rep["1"])],
        ["Obfuskuota", *report_to_record(rep["2"])],
        ["Svertinis vidurkis", *report_to_record(rep["weighted avg"])],
    ]

    return pd.DataFrame(
        lst, columns=("Klasė", "Preciziškumas", "Atkūrimas", "F1")
    )
#%% 2x2
names = {"normal": "normal", "lime": "synthesis", "lime_cat": "lime"}
reports = [(name, f"../../metrics/{name}.json") for name in names.keys()]
for name, report in reports:
    data = json.load(open(report, "r", encoding="utf-8"))
    df = report_to_df_2d(data)
    acc = data["accuracy"]
    print(f"Acc {names[name]}: {acc}")
    df.to_csv(f"{names[name]}_2x2.csv", index=False, float_format="%.3f")
    
#%% 3x3
names = {"lime_obf": "synthesis", "lime_cat_obf": "lime"}
reports = [(name, f"../../metrics/{name}.json") for name in names.keys()]
for name, report in reports:
    data = json.load(open(report, "r", encoding="utf-8"))
    df = report_to_df_3d(data)
    acc = data["accuracy"]
    print(f"Acc {names[name]}: {acc}")
    df.to_csv(f"{names[name]}_3x3.csv", index=False, float_format="%.3f")
# %%
