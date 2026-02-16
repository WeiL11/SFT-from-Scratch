# Fetch & Clean Vast.ai Server Data

Run these when your vast.ai instance is **running**.

## 1. Start your vast.ai instance
Ensure the instance is running in the vast.ai dashboard. Note the SSH host and port (e.g. `root@ssh7.vast.ai`, port `12545`).

## 2. Set environment variables (keep these private, do not commit)
```bash
export VAST_SSH_HOST="root@YOUR_VAST_HOST"   # e.g. root@ssh7.vast.ai
export VAST_SSH_PORT="YOUR_SSH_PORT"         # e.g. 12545
export LOCAL_DEST="$HOME/remote_backup"      # where to save fetched data
```

## 3. List what's on the server
```bash
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "ls -la /root/ && du -sh /root/*"
```

## 4. Fetch (download) data
```bash
mkdir -p "$LOCAL_DEST"
scp -P $VAST_SSH_PORT -r $VAST_SSH_HOST:/root/experiments "$LOCAL_DEST/"
scp -P $VAST_SSH_PORT -r $VAST_SSH_HOST:/root/Alignment "$LOCAL_DEST/" 2>/dev/null || true
scp -P $VAST_SSH_PORT $VAST_SSH_HOST:/root/gsm8k.json "$LOCAL_DEST/" 2>/dev/null || true
```

## 5. Clean the server (free space)
```bash
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "
  rm -rf /root/experiments
  rm -rf /root/.cache/huggingface
  rm -rf /root/.cache/pip
  find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  df -h /
"
```

## Or run the script
```bash
chmod +x fetch_and_clean_remote.sh
VAST_SSH_HOST=root@YOUR_HOST VAST_SSH_PORT=YOUR_PORT LOCAL_DEST=$HOME/remote_backup ./fetch_and_clean_remote.sh
```

## Note
- **experiments/** = Only `final_model/` (~3GB) + `run_info.json` (which model is better) per run
- **experiments/tune_results/** = `tune_summary_*.json` (which config is best)
- **.cache/huggingface** = downloaded models (~3–6GB). Deleting = re-download on next run
