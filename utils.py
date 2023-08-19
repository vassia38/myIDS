import pandas as pd
import numpy as np
from matplotlib import pyplot
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from config import *

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
        benign_df.to_csv(new_file_path + "_Benign.csv", header=bHeader, mode="a", index=False)
        mal_df.to_csv(new_file_path + "_Malicious.csv", header=bHeader, mode="a", index=False)

        bHeader = False


def split_and_export_datasets_by_label(filepath: str, chunk_size = 1000):
    new_file_path = filepath.split(".csv")[0].split("_")[-2]
    
    bHeader = False
    counter = 0
    for chunk in pd.read_csv(filepath, chunksize = chunk_size):
        print("chunk " + str(counter))
        counter += chunk_size

        df = pd.DataFrame(chunk)
        for type in df["Attack"].unique():
            filtered = df[df["Attack"] == type]
            fp = new_file_path + "_" + type + ".csv"
            bHeader = False
            if not os.path.exists(fp):
                bHeader = True
            filtered.to_csv(fp, header=bHeader, mode="a", index=False)


def get_sparsity_ratio(X: pd.DataFrame):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])


def load_data(benigndata_file: str, maliciousdata_folder: str,
              benign_nrows=10000, benign_skip_n=0,
              malicious_nrows=1000, malicious_skip_n=0,
              binary_classif=False) -> pd.DataFrame:
    
    df_b = pd.read_csv(benigndata_file, nrows=benign_nrows, skiprows=range(1, benign_skip_n))
    df = pd.concat([df_b,], ignore_index=True)

    for basename in os.listdir(maliciousdata_folder):
        if basename.endswith('.csv'):
            pathname = os.path.join(maliciousdata_folder, basename)
            df_m = pd.read_csv(pathname, nrows=malicious_nrows, skiprows=range(1, malicious_skip_n))
            df = pd.concat([df, df_m], ignore_index=True)
    
    df = df.sample(frac=1).reset_index(drop=True)
    if binary_classif:
        # drop only the following columns:
        # df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Attack"])
        # keep only the following columns:
        df = df[["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Label"]]
    else:
        # drop only the following columns:
        # df.drop(inplace=True, columns=["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "Dataset", "Label"])
        # keep only the following columns:
        df = df[["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Attack"]]
    
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
    labels_mapping = {}
    for i in range(len(LABELS)):
        labels_mapping[i] = LABELS[i]
    return encoded_labels, labels_mapping


def load_model(filename="RF_classifier.sav") -> RandomForestClassifier:
    return joblib.load(filename)


def save_model(model: RandomForestClassifier, filename="RF_classifier.sav"):
    joblib.dump(model, filename)

# filepath = os.path.abspath(".") + "\\" + DATASET_NAME + ".csv"
# split_and_export_dataset_0and1(filepath, 100000)
# filepath = os.path.abspath(".") + "\\" + DATASET_NAME + "_Malicious" + ".csv"
# split_and_export_datasets_by_label(filepath)