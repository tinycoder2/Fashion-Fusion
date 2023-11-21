import os
import pandas as pd

root_dir = "./dataset"

dataset = []

for dir, subdir, files in os.walk(root_dir):
    for file in files:
        label = dir.split("\\")[1]
        filepath = os.path.join(dir, file)
        dataset.append([filepath, label])

df = pd.DataFrame(data=dataset, columns=["filepath", "label"])
df.to_csv("dataset8.csv", index=False)
