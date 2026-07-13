import pandas as pd

from utils.iemocap_parser import IEMOCAPParser

root = "data/IEMOCAP/IEMOCAP_full_release"

parser = IEMOCAPParser(root)

rows = parser.parse()

df = pd.DataFrame(rows)

print(df.head())
print("Samples:", len(df))

df.to_csv(
    "data/iemocap.csv",
    index=False
)