#!/bin/bash
# Sync collected data from EC2 back to local machine
# Usage: ./sync-from-ec2.sh [EC2_IP_ADDRESS]

set -e

# Defaults
DEFAULT_EC2_IP="3.85.167.225"
SSH_KEY="$HOME/.ssh/ticket-predictor-key.pem"

EC2_IP="${1:-$DEFAULT_EC2_IP}"
EC2_USER="ubuntu"
EC2_LISTINGS_PATH="/home/ubuntu/ticket-price-predictor/data/raw/listings"
EC2_SNAPSHOTS_PATH="/home/ubuntu/ticket-price-predictor/data/raw/snapshots"
LOCAL_LISTINGS_PATH="./data/raw/listings"
LOCAL_SNAPSHOTS_PATH="./data/raw/snapshots"

echo "=== Syncing data from EC2 ==="
echo "EC2: $EC2_USER@$EC2_IP"
echo ""

# Create local directories if they don't exist
mkdir -p "$LOCAL_LISTINGS_PATH"
mkdir -p "$LOCAL_SNAPSHOTS_PATH"

# Sync listings
echo "--- Syncing listings ---"
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    "$EC2_USER@$EC2_IP:$EC2_LISTINGS_PATH/" \
    "$LOCAL_LISTINGS_PATH/"

# Sync snapshots
echo ""
echo "--- Syncing snapshots ---"
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    "$EC2_USER@$EC2_IP:$EC2_SNAPSHOTS_PATH/" \
    "$LOCAL_SNAPSHOTS_PATH/" 2>/dev/null || echo "No snapshots directory on EC2 yet"

echo ""
echo "=== Sync Complete ==="

# Show stats
cd "$(dirname "$0")/.."
python3 -c "
from pathlib import Path
import pyarrow.parquet as pq

for name in ['listings', 'snapshots']:
    data_dir = Path(f'data/raw/{name}')
    total = 0
    for f in data_dir.rglob('*.parquet'):
        table = pq.read_table(f)
        total += len(table)
    print(f'Total {name} in local database: {total:,}')
" 2>/dev/null || echo "Run 'python scripts/check_data.py' to see stats"
