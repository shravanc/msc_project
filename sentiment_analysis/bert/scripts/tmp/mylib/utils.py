import os
import pandas as pd

def load_datasets(train_base_dir, valid_base_dir):

    train_df = pd.DataFrame()
    for name in os.listdir(train_base_dir):
        file_path = os.path.join(train_base_dir, name)
        train_df = pd.concat([train_df,
                             pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True)


        valid_df = pd.DataFrame()
        for name in os.listdir(valid_base_dir):
            file_path = os.path.join(valid_base_dir, name)
            valid_df = pd.concat([valid_df,
                                  pd.read_csv(file_path, sep=',', names=['sentences', 'polarity'])],
                                 ignore_index=True)

    return train_df, valid_df



