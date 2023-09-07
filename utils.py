import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from config import *
from scp import SCPClient
from paramiko import SSHClient

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
            
            if basename.lower().find("worms") != -1:
                if malicious_skip_n == 0:
                    df_m = pd.read_csv(pathname, nrows=100, skiprows=range(1, 0))
                else:
                    df_m = pd.read_csv(pathname, nrows=64, skiprows=range(1, 100))
            else:
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

def decode_labels(arr: np.array):
    labels_mapping = {}
    for i in range(len(LABELS)):
        labels_mapping[i] = LABELS[i]
    
    decoded_labels = [labels_mapping[label] for label in arr]
    labels_mapping = {}

    return decoded_labels, labels_mapping


def make_flags_byte_from_string(s: str) -> int:
    n = 0
    fl = 0
    for i in range(len(s), 0, -1):
        if s[i-1] != '.':
            fl += pow(2,n)
        n += 1
    return fl


def from_seconds_to_millis(ts: float) -> int:
    return int(ts * 1000)


def load_model(filename="RF_classifier.sav") -> RandomForestClassifier:
    return joblib.load(filename)


def save_model(model: RandomForestClassifier, filename="RF_classifier.sav"):
    joblib.dump(model, filename)

def get_RandomForestHyperparams_Grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                }
    
    return random_grid

def SCPGet_nf_dump(date: str, time1: str = None, time2: str = None):
    """
    Returns: -
    CSV file of all netflow data dumped from a certain period
    Value return has type string
    If only date is specified, then dump all netflow from that certain day
    If time1 is specified, dump all netflow starting from that time
    If time1 and time2 are specified, dump netflow only from that time window
     
    Parameter date: expr that may be any of:
        YYYY/MM/DD
        YYYY/MM
    Parameter time1 (optional): <HHmm[ss]>
    Parameter time2 (optional): <HHmm[ss]>
    Preconditions: time1,time2 must be multiples of 5 (ex. 1800, 1805, 0900,125030)
    """
    if len(date) != len(r'YYYY/MM/DD') and len(date) != len(r'YYYY/MM'):
        return False
    if time1 != None and len(date.split('/')) != 3:
        return False
    
    date_no_slashes = ''
    for s in date.split('/'):
        date_no_slashes += s
    try:
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect('192.168.2.1', username='root', password='rootoot')

            cmd = NFDUMP + r" -R /var/log/" 
            cmd += date
            cmd += '/' + NFCAPD + '.' + date_no_slashes + time1 if time1 is not None else ''
            cmd += ':' + NFCAPD + '.' + date_no_slashes + time2 if time1 is not None and time2 is not None else ''
            cmd += r" -N -B -q -o " + DUMP_FORMAT
            cmd += r" | cat > /out.csv"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()          # Blocking call
            if exit_status == 0:
                print ("Dump created on host machine")
            else:
                raise Exception("Error on host machine ", exit_status)
            print(cmd)
            with SCPClient(ssh.get_transport()) as scp:
                scp.get('/out.csv')
        return True
    except Exception as e:
        print(e)
        return False

# filepath = os.path.abspath(".") + "\\" + DATASET_NAME + ".csv"
# split_and_export_dataset_0and1(filepath, 100000)
# filepath = os.path.abspath(".") + "\\" + DATASET_NAME + "_Malicious" + ".csv"
# split_and_export_datasets_by_label(filepath)