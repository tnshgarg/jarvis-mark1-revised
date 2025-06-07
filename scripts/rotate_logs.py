# scripts/rotate_logs.py
import os, shutil
from datetime import datetime

DATA_DIR = "./data/tasks/"
LOG_FILE = "history.json"
BACKUP_DIR = "./data/backups/"

def rotate_log():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"{timestamp}_{LOG_FILE}"
    src_path = os.path.join(DATA_DIR, LOG_FILE)
    dst_path = os.path.join(BACKUP_DIR, backup_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        open(src_path, "w").close()  # Empty current log
        print(f"[✓] Rotated log to {dst_path}")
    else:
        print("[!] No history.json file found to rotate.")

if __name__ == "__main__":
    rotate_log()
