from pathlib import Path

DUMP_FORMAT = r"'fmt:%sa,%sp,%da,%dp,%pr,%ibyt,%ipkt,%obyt,%opkt,%flg,%td,%bpp,%pps'"
NFCAPD = 'nfcapd'
NFDUMP = 'nfdump'

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATASET_NAME = "NF-UQ-NIDS-v2"
LABELS = [line.strip() for line in (ROOT_DIR / 'labels.txt').read_text().splitlines() if line.strip()]