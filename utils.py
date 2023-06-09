import pandas as pd
import numpy as np
from matplotlib import pyplot
import os, joblib
from sklearn.ensemble import RandomForestClassifier

DATASET_NAME = "NF-UQ-NIDS-v2"
LABELS = []
for line in open('labels.txt', 'r').readlines():
    LABELS.append(line.strip())


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

def load_data(benign_datafile: str, malicious_datafile: str,
              benign_nrows=10000, benign_skip_n=0,
              malicious_nrows=2500, malicious_skip_n=0) -> pd.DataFrame:
    df_b = pd.read_csv(benign_datafile, nrows=benign_nrows, skiprows=range(1, benign_skip_n))
    df_m = pd.read_csv(malicious_datafile, nrows=malicious_nrows, skiprows=range(1, malicious_skip_n))
    df = pd.concat([df_b, df_m], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Label"])
    return clean_data(df)

def clean_data(df: pd.DataFrame):
    int_columns = df.select_dtypes(include='float64').columns
    for column in int_columns:
        df[column] = np.float32(df[column].values)
        #print(np.any(df[column].isin([np.nan, np.inf, -np.inf])))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()

def encode_labels(arr: np.array):
    labels_mapping = {}
    for i in range(len(LABELS)):
        labels_mapping[LABELS[i]] = i
    
    encoded_labels = [labels_mapping[label] for label in arr]
    return encoded_labels, labels_mapping



def load_model(filename="RF_classifier.sav") -> RandomForestClassifier:
    return joblib.load(filename)

def save_model(model: RandomForestClassifier, filename="RF_classifier.sav"):
    joblib.dump(model, filename)

# filepath = os.path.abspath(".") + "\\" + DATASET_NAME + ".csv"
# split_and_export_dataset_0and1(filepath, 100000)