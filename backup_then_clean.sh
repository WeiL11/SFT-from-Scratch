#!/bin/bash
# Backup remote data to a TIMESTAMPED directory (never overwrites old saves), then clean the remote.
# Usage: Set VAST_SSH_HOST, VAST_SSH_PORT, LOCAL_BASE env vars, then ./backup_then_clean.sh
# Example: VAST_SSH_HOST=root@ssh7.vast.ai VAST_SSH_PORT=12545 ./backup_then_clean.sh

SSH_HOST="${VAST_SSH_HOST:-root@YOUR_VAST_HOST}"
SSH_PORT="${VAST_SSH_PORT:-YOUR_SSH_PORT}"
LOCAL_BASE="${LOCAL_BASE:-$HOME/Downloads/Alignment/remote_backup}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_DEST="${LOCAL_BASE}/backup_${TIMESTAMP}"

set -e

echo "=== Backup then clean (data will NOT overwrite existing backups) ==="
echo "Backup destination: $LOCAL_DEST"
echo ""

echo "=== 1. Listing remote data ==="
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "
  echo 'Disk usage:'
  df -h /
  du -sh /root/* 2>/dev/null || true
"

echo ""
echo "=== 2. Fetching to $LOCAL_DEST (timestamped - safe) ==="
mkdir -p "$LOCAL_DEST"

for dir in experiments Alignment .cache; do
  if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "test -d /root/$dir" 2>/dev/null; then
    echo "Copying /root/$dir ..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r "$SSH_HOST:/root/$dir" "$LOCAL_DEST/" 2>/dev/null || true
  fi
done

if ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "test -d /workspace" 2>/dev/null; then
  echo "Copying /workspace ..."
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r "$SSH_HOST:/workspace" "$LOCAL_DEST/" 2>/dev/null || true
fi

# Also grab loose files
for f in gsm8k.json; do
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST:/root/$f" "$LOCAL_DEST/" 2>/dev/null || true
done

echo ""
echo "=== 3. Cleaning remote (freeing space) ==="
ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" "
  echo 'Removing experiments/...'
  rm -rf /root/experiments 2>/dev/null || true
  echo 'Removing HuggingFace cache...'
  rm -rf /root/.cache/huggingface 2>/dev/null || true
  echo 'Removing pip cache...'
  rm -rf /root/.cache/pip 2>/dev/null || true
  echo 'Removing __pycache__...'
  find /root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  echo ''
  echo 'Remaining disk usage:'
  df -h /
"

echo ""
echo "Done. Backup saved to: $LOCAL_DEST"
echo "Previous backups in: $LOCAL_BASE/"
