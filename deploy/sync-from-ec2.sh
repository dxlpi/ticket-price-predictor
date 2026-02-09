#!/bin/bash
# Sync collected data from EC2 back to local machine
# Usage: ./sync-from-ec2.sh [EC2_IP_ADDRESS]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <EC2_IP_ADDRESS>"
    echo "Example: $0 54.123.45.67"
    exit 1
fi

EC2_IP=$1
EC2_USER="ubuntu"
EC2_PATH="/home/ubuntu/ticket-price-predictor/data/raw/listings"
LOCAL_PATH="./data/raw/listings"

echo "=== Syncing data from EC2 ==="
echo "EC2: $EC2_USER@$EC2_IP:$EC2_PATH"
echo "Local: $LOCAL_PATH"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Sync using rsync (preserves partitions and only transfers new files)
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$EC2_USER@$EC2_IP:$EC2_PATH/" \
    "$LOCAL_PATH/"

echo ""
echo "=== Sync Complete ==="

# Show stats
cd "$(dirname "$0")/.."
python3 -c "
from pathlib import Path
import pyarrow.parquet as pq

listings_dir = Path('data/raw/listings')
total = 0
for f in listings_dir.rglob('*.parquet'):
    table = pq.read_table(f)
    total += len(table)
print(f'Total listings in local database: {total:,}')
" 2>/dev/null || echo "Run 'python scripts/check_data.py' to see stats"
