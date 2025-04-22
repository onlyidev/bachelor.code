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
    
#%% Accuracy collector
def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))
class AccCollector:
    def __init__(self):
        self.acc = {}

    def add(self, name, acc):
        self.acc[name] = acc

    def print(self):
        for name, acc in self.acc.items():
            print(f"Acc {name}: {round(acc, 3)}")

    def latex(self):
        return ["\\def\\acc{0}{{\\num{{{1}}}}}".format(to_camel_case(name), round(acc, 3)) for name, acc in self.acc.items()]
            
acc = AccCollector()
#%% 2x2
names = {"normal": "normal_2x2", "lime": "synthesis_2x2", "lime_cat": "lime_2x2"}
reports = [(name, f"../../metrics/{name}.json") for name in names.keys()]
for name, report in reports:
    data = json.load(open(report, "r", encoding="utf-8"))
    df = report_to_df_2d(data)
    acc.add(name, data["accuracy"])
    df.to_csv(f"{names[name]}.csv", index=False, float_format="%.3f")
    
#%% 3x3
names = {"lime_obf": "synthesis_3x3", "lime_cat_obf": "lime_3x3"}
reports = [(name, f"../../metrics/{name}.json") for name in names.keys()]
for name, report in reports:
    data = json.load(open(report, "r", encoding="utf-8"))
    df = report_to_df_3d(data)
    acc.add(name, data["accuracy"])
    df.to_csv(f"{names[name]}.csv", index=False, float_format="%.3f")
# %%
acc.print()
with open("acc.tex", "w") as f:
    f.writelines([f"{str}\n" for str in acc.latex()])