from scp import SCPClient
from paramiko import SSHClient
from config import *
import utils

def SCPGet_nf_dump(date: str, time1: str = None, time2: str = None):
    """
    Returns: -
    CSV file of all netflow data dumped from a certain period
    Value return has type string
    If only date is specified, then dump all netflow from that certain day
    If time1 is specified, dump netflow from a 5-min window
    If time1 and time2 are specified, dump netflow from that time window
     
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
            cmd += r" -N -B -o " + DUMP_FORMAT
            cmd += r" | cat > /out.csv"
            print(cmd)
            ssh.exec_command(cmd)
            with SCPClient(ssh.get_transport()) as scp:
                scp.get('/out.csv')
        return True
    except:
        pass
    return False

if SCPGet_nf_dump('2023/08/18', '1840'):
    rfc_bin = utils.load_model("RF_SMOTE_bin.sav")
    rfc_multi = utils.load_model("RF_SMOTE_multi.sav")