# Fetch & Clean Vast.ai Server Data

Run these when your vast.ai instance is **running**.

## 0. Find what's using disk (run first if 150GB is full)

```bash
# Overall usage
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "df -h /"

# Top-level folders
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "du -sh /root/* 2>/dev/null | sort -hr | head -20"

# experiments/ breakdown (often the culprit)
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "du -sh /root/experiments/* 2>/dev/null | sort -hr"

# Check for old checkpoints (3GB each - huge!)
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "find /root/experiments -name 'checkpoint_step_*' -type d 2>/dev/null | head -20"

# .cache breakdown
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "du -sh /root/.cache/* 2>/dev/null | sort -hr"
```

**Common 150GB culprits:** `checkpoint_step_*/` (~3GB each), `final_model/` (~3GB each), `.cache/huggingface/`

## 1. Start your vast.ai instance
Ensure the instance is running in the vast.ai dashboard. Note the SSH host and port (e.g. `root@ssh7.vast.ai`, port `12545`).

## 2. Set environment variables (keep these private, do not commit)
```bash
export VAST_SSH_HOST="root@YOUR_HOST"   # e.g. root@ssh7.vast.ai
export VAST_SSH_PORT="YOUR_PORT"        # e.g. 12545
export LOCAL_DEST="$HOME/Downloads/Alignment/remote_backup"

# Full SSH (Jupyter): ssh -p $VAST_SSH_PORT $VAST_SSH_HOST -L 8080:localhost:8080
```

## 3. List what's on the server
```bash
ssh -o StrictHostKeyChecking=no -p $VAST_SSH_PORT $VAST_SSH_HOST "ls -la /root/ && du -sh /root/*"
```

## 4. Fetch (download) data
```bash
mkdir -p "$LOCAL_DEST"
scp -P $VAST_SSH_PORT -r $VAST_SSH_HOST:/root/experiments "$LOCAL_DEST/"
scp -P $VAST_SSH_PORT -r $VAST_SSH_HOST:/root/Alignment "$LOCAL_DEST/" 2>/dev/null || true
scp -P $VAST_SSH_PORT $VAST_SSH_HOST:/root/gsm8k.json "$LOCAL_DEST/" 2>/dev/null || true
```

## 5. Clean the server (free space)

**Option A: Nuclear (removes everything – fetch first!)**
```bash
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "
  rm -rf /root/experiments
  rm -rf /root/.cache/huggingface
  rm -rf /root/.cache/pip
  find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  df -h /
"
```

**Option B: Keep final_model, remove only checkpoints (saves ~3GB × N per run)**
```bash
ssh -p $VAST_SSH_PORT $VAST_SSH_HOST "
  find /root/experiments -type d -name 'checkpoint_step_*' -exec rm -rf {} + 2>/dev/null
  find /root/experiments -name 'samples_step_*.json' -delete 2>/dev/null
  rm -rf /root/.cache/huggingface
  rm -rf /root/.cache/pip
  df -h /
"
```

## Or run the script
```bash
chmod +x fetch_and_clean_remote.sh
VAST_SSH_HOST=root@YOUR_HOST VAST_SSH_PORT=YOUR_PORT LOCAL_DEST=$HOME/remote_backup ./fetch_and_clean_remote.sh
```

## Disk usage summary

| Item | Size |
|------|------|
| `final_model/` (per run) | ~3 G |
| `checkpoint_step_*/` (each) | ~3 G |
| `.cache/huggingface/` | ~3–6 G |
| `experiments/` (3 tuning runs) | ~9 G |
| `experiments/` (with checkpoints, ~37 per run) | ~120 G |
| `.cache/pip/` | ~0.1–0.5 G |
| `tune_results/` (run_*.json) | &lt;0.01 G |

## Note
- **experiments/** = Only `final_model/` (~3 G) + `run_info.json` (which model is better) per run
- **experiments/tune_results/** = `tune_summary_*.json` (which config is best)
- **.cache/huggingface** = downloaded models (~3–6 G). Deleting = re-download on next run
