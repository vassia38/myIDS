import pandas as pd
import numpy as np
from matplotlib import pyplot
import os

DATASET_NAME = "NF-UQ-NIDS-v2"

def print_feature_distribution(df: pd.DataFrame):
    for col in df.columns:
        # counts, bins = np.histogram(df[col].values)
        pyplot.hist(df[col].values, label=str(df[col].name))
        pyplot.show()

def split_and_export_dataset_0and1(filepath: str, chunk_size = 1):
    new_file_path = filepath.split(".csv")[0]
    if os.path.exists(new_file_path + "_benign.csv"):
        return
    
    bHeader = True
    counter = 0
    for chunk in pd.read_csv(filepath, chunksize = chunk_size):
        print("chunk " + str(counter))
        counter += chunk_size
        
        df = pd.DataFrame(chunk)
        benign_df = df[df["Label"] == 0]
        mal_df = df[df["Label"] == 1]
        benign_df.to_csv(new_file_path + "_benign.csv", header=bHeader, mode="a", index=False)
        mal_df.to_csv(new_file_path + "_malicious.csv", header=bHeader, mode="a", index=False)

        bHeader = False

def get_sparsity_ratio(X: pd.DataFrame):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def load_data(benign_datafile: str, malicious_datafile: str, benign_nrows = 1000, malicious_nrows = 2500) -> pd.DataFrame:
    df_b = pd.read_csv(benign_datafile, nrows=benign_nrows)
    df_m = pd.read_csv(malicious_datafile, nrows=malicious_nrows)
    df = pd.concat([df_b, df_m], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Attack"])
    return df
    
        
#filepath = os.path.abspath(".") + "\\" + DATASET_NAME + ".csv"
#split_and_export_dataset_0and1(filepath, 100000)