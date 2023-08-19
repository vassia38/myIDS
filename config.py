DUMP_FORMAT = r"'fmt:%sa,%sp,%da,%dp,%pr,%ibyt,%ipkt,%obyt,%opkt,%flg,%td,%bpp,%pps'"
NFCAPD = 'nfcapd'
NFDUMP = 'nfdump'

DATASET_NAME = "NF-UQ-NIDS-v2"
LABELS = []
for line in open('labels.txt', 'r').readlines():
    LABELS.append(line.strip())