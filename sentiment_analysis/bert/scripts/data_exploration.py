import matplotlib.pyplot as plt
import pandas as pd

import os
train_base_dir = "/home/shravan/Downloads/train/"
valid_base_dir = "/home/shravan/Downloads/valid/"
#train_base_dir= "/home/shravan/Downloads/valid/"


def load_datasets():
    train_df = pd.DataFrame()
    for name in os.listdir(train_base_dir):
        file_path = os.path.join(train_base_dir, name)
        train_df = pd.concat([train_df,
                              pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True
                             )


    return train_df


df = load_datasets()

print(df.head())
print(len(df))
df['polarity'].hist(bins=50)

plt.show()
