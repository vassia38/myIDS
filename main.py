from scp import SCPClient
from paramiko import SSHClient
from config import *
import utils
import pandas as pd
import argparse
from datetime import date
import sys

parser = argparse.ArgumentParser(
                    prog='myIDS',
                    description='Get Netflow v9 dump from router and analyze')
parser.add_argument('-d', '--date', default=[date.today().strftime(r"%Y/%m/%d")], nargs=1)
parser.add_argument('-t', '--time')


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
    Parameter time1 (optional): <HHmm>
    Parameter time2 (optional): <HHmm>
    Preconditions: time1,time2 must be multiples of 5 (ex. 1800, 1805, 0900)
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

def main():
    args = parser.parse_args()
    time1 = None
    time2 = None
    if args.time is not None:
        timestamps = args.time.split("-")
        time1 = timestamps[0]
        if len(timestamps) > 1:
            time2 = timestamps[1]
    print("args are ", args.date[0], time1, time2)

    if not SCPGet_nf_dump(args.date[0], time1, time2):
        return 1
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
    except pd.errors.EmptyDataError as e:
        print(e)

    return 0


if __name__ == '__main__':
    sys.exit(main())