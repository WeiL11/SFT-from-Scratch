#!/bin/bash
# Fetch data from vast.ai server and clean remote to free space
# Usage: Set VAST_SSH_HOST, VAST_SSH_PORT, LOCAL_DEST env vars, then ./fetch_and_clean_remote.sh
# Example: VAST_SSH_HOST=root@ssh7.vast.ai VAST_SSH_PORT=12545 LOCAL_DEST=$HOME/remote_backup ./fetch_and_clean_remote.sh

SSH_HOST="${VAST_SSH_HOST:-root@YOUR_VAST_HOST}"
SSH_PORT="${VAST_SSH_PORT:-YOUR_SSH_PORT}"
REMOTE_BASE="/root"
LOCAL_DEST="${LOCAL_DEST:-$HOME/remote_backup}"

set -e

echo "=== 1. Listing remote data (run this when server is up) ==="
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "
  echo 'Contents of /root:'
  ls -la /root/
  echo ''
  echo 'Disk usage:'
  du -sh /root/* 2>/dev/null || true
  echo ''
  echo 'Looking for experiments, checkpoints, models:'
  find /root -maxdepth 4 -type d \( -name 'experiments' -o -name 'checkpoint*' -o -name '*model*' -o -name '.cache' \) 2>/dev/null | head -20
"

echo ""
echo "=== 2. Fetching data to $LOCAL_DEST ==="
mkdir -p "$LOCAL_DEST"

# Fetch common locations
for dir in experiments Alignment .cache; do
  if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "test -d /root/$dir" 2>/dev/null; then
    echo "Copying /root/$dir ..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r "$SSH_HOST:/root/$dir" "$LOCAL_DEST/" 2>/dev/null || true
  fi
done

# Also try /workspace if it exists
if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "test -d /workspace" 2>/dev/null; then
  echo "Copying /workspace ..."
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r "$SSH_HOST:/workspace" "$LOCAL_DEST/" 2>/dev/null || true
fi

echo ""
echo "=== 3. Cleaning remote (freeing space) ==="
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "
  echo 'Removing experiments/ (checkpoints, samples)...'
  rm -rf /root/experiments 2>/dev/null || true
  
  echo 'Removing HuggingFace cache...'
  rm -rf /root/.cache/huggingface 2>/dev/null || true
  
  echo 'Removing pip cache...'
  rm -rf /root/.cache/pip 2>/dev/null || true
  
  echo 'Removing __pycache__...'
  find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  
  echo 'Remaining disk usage:'
  df -h /
  du -sh /root/* 2>/dev/null || true
"

echo ""
echo "Done. Data saved to $LOCAL_DEST"
