from config import *
import utils
import pandas as pd
import argparse
from datetime import date, datetime
import sys
from time import sleep

parser = argparse.ArgumentParser(
                    prog='myIDS',
                    description='Get Netflow v9 dump from router and analyze')
parser.add_argument('-d', '--date')
parser.add_argument('-t', '--time')
parser.add_argument('-D', '--daemon', default=0)

def prepare_dump(df: pd.DataFrame) -> pd.DataFrame:
    """
    dataframe columns:
    0            1       2            3       4      5        6       7         8        9      10        11   12
    Src IP Addr  Src Pt  Dst IP Addr  Dst Pt  Proto  In Byte  In Pkt  Out Byte  Out Pkt  Flags  Duration  Bpp  pps
    """
    df[9] = df[9].apply(utils.make_flags_byte_from_string)
    df[10] = df[10].apply(utils.from_seconds_to_millis)
    df.drop(columns=[0,2,11,12], inplace=True)
    #print(df)
    return df

def analyze():
    try:
        rfc = utils.load_model("RF_SMOTE_multi.sav")
        df = prepare_dump(pd.read_csv("out.csv", header=None))
        X = df.values
        y_pred = rfc.predict(X)
        print("Predictions:")
        print(y_pred)

        mal = 0
        for v in y_pred:
            if v != 0:
                mal += 1
        print(str(mal) + " out of " + str(len(y_pred)) + " considered malware")

        dump_file = "dump." + date.today().strftime(r"%Y%m%d") + datetime.today().strftime(r"%H%M%S") + ".csv"
        decoded_pred, mapping = utils.decode_labels(y_pred)
        df['Label'] = decoded_pred
        df.to_csv("./dump/" + dump_file)

    except pd.errors.EmptyDataError as e:
        print(e)

def main():
    args = parser.parse_args()
    print("args are ", args)
    if args.daemon == 0:
        if args.date == None:
            args.date = date.today().strftime(r"%Y/%m/%d")
        time1 = None
        time2 = None
        if args.time is not None:
            timestamps = args.time.split("-")
            time1 = timestamps[0]
            if len(timestamps) > 1:
                time2 = timestamps[1]
        print("Get log for ", args.date, time1, time2)

        if not utils.SCPGet_nf_dump(args.date, time1, time2):
                return 1
        analyze()

    return 0


if __name__ == '__main__':
    sys.exit(main())