# AWS EC2 Free Tier Setup Guide

Complete guide to deploy the ticket price predictor on AWS EC2 free tier.

## Prerequisites

- AWS account (with free tier eligibility)
- AWS CLI installed and configured locally
- SSH key pair for EC2 access

## Step 1: Configure AWS CLI

```bash
aws configure
# AWS Access Key ID: [Your key]
# AWS Secret Access Key: [Your secret]
# Default region: us-east-1 (recommended for free tier)
# Default output format: json
```

## Step 2: Create EC2 Instance

### Option A: Using AWS CLI (Automated)

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name ticket-predictor-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/ticket-predictor-key.pem

chmod 400 ~/.ssh/ticket-predictor-key.pem

# Create security group
aws ec2 create-security-group \
    --group-name ticket-predictor-sg \
    --description "Security group for ticket predictor"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-name ticket-predictor-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# Launch t3.micro instance (free tier)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name ticket-predictor-key \
    --security-groups ticket-predictor-sg \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ticket-predictor}]'
```

### Option B: Using AWS Console (Manual)

1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2)
2. Click "Launch Instance"
3. Choose:
   - **Name:** ticket-predictor
   - **AMI:** Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance type:** t3.micro (Free tier eligible)
   - **Key pair:** Create new or use existing
   - **Security group:** Allow SSH (port 22) from your IP
4. Click "Launch Instance"

## Step 3: Connect to EC2 Instance

```bash
# Get instance public IP
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ticket-predictor" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text

# SSH into instance
ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@[EC2_PUBLIC_IP]
```

## Step 4: Setup EC2 Instance

On your **local machine**, copy setup scripts to EC2:

```bash
scp -i ~/.ssh/ticket-predictor-key.pem \
    deploy/ec2-setup.sh \
    deploy/ec2-service.sh \
    ubuntu@[EC2_PUBLIC_IP]:~/
```

On the **EC2 instance** (via SSH):

```bash
chmod +x ec2-setup.sh ec2-service.sh
./ec2-setup.sh

# This will:
# - Install Python 3.11
# - Install Playwright and dependencies
# - Create virtual environment
# - Install packages (pydantic, pyarrow, pandas, playwright, ytmusicapi, httpx, etc.)
# - Set up project directories
```

## Step 5: Deploy Project Files

On your **local machine**:

```bash
# One-command deploy (syncs code, .env, installs packages, restarts timer)
./deploy/deploy-to-ec2.sh [EC2_PUBLIC_IP]
```

Or manually:

```bash
# Rsync source code
rsync -avz --progress \
    -e "ssh -i ~/.ssh/ticket-predictor-key.pem" \
    --exclude 'venv/' --exclude '.venv/' --exclude '.git/' \
    --exclude 'data/' --exclude '__pycache__/' --exclude '.omc/' \
    --exclude 'logs/' --exclude '.mypy_cache/' --exclude '*.pyc' \
    src/ scripts/ pyproject.toml \
    ubuntu@[EC2_PUBLIC_IP]:~/ticket-price-predictor/

# Transfer .env with API keys
scp -i ~/.ssh/ticket-predictor-key.pem \
    .env ubuntu@[EC2_PUBLIC_IP]:~/ticket-price-predictor/.env
```

## Step 6: Install Service

On the **EC2 instance**:

```bash
# Install systemd service and timer
./ec2-service.sh

# Check timer status
sudo systemctl status ticket-monitor.timer

# View logs
tail -f ~/ticket-price-predictor/logs/monitor.log
```

## Step 7: Sync Data Back to Local

On your **local machine** (run daily or as needed):

```bash
# Sync data from EC2 (IP defaults to 3.85.167.225)
./deploy/sync-from-ec2.sh

# Or set up automated daily sync (crontab -e):
0 8 * * * cd /Users/heather/ticket-price-predictor && ./deploy/sync-from-ec2.sh >> logs/sync.log 2>&1
```

## Environment Variables

The `.env` file needs:
```
TICKETMASTER_API_KEY=your_key_here
LASTFM_API_KEY=your_key_here
# YouTube Music uses ytmusicapi (no API key needed)
```

## Management Commands

### On EC2 Instance

```bash
# Check timer/service status
sudo systemctl status ticket-monitor.timer
sudo systemctl status ticket-monitor.service

# View logs
sudo journalctl -u ticket-monitor -f
tail -f ~/ticket-price-predictor/logs/monitor.log

# Manual test run
cd ~/ticket-price-predictor
source venv/bin/activate
python scripts/monitor_popular.py

# Stop/start timer
sudo systemctl stop ticket-monitor.timer
sudo systemctl start ticket-monitor.timer
```

### On Local Machine

```bash
# Deploy code updates
./deploy/deploy-to-ec2.sh

# Sync data from EC2
./deploy/sync-from-ec2.sh

# SSH into EC2
ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@3.85.167.225

# Check EC2 instance status
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ticket-predictor" \
    --query "Reservations[0].Instances[0].State.Name"

# Stop/start EC2 instance
aws ec2 stop-instances --instance-ids i-0e0fc71a207fedc21
aws ec2 start-instances --instance-ids i-0e0fc71a207fedc21
```

## Cost Optimization

**Free Tier Limits:**
- 750 hours/month of t3.micro (24/7 for one instance)
- 30 GB of EBS storage
- 15 GB of bandwidth

**Tips:**
- Keep instance running 24/7 to stay within free tier
- Monitor usage at: https://console.aws.amazon.com/billing/home#/freetier
- Set up billing alerts for when you exceed free tier

## Troubleshooting

### Service not running
```bash
sudo journalctl -u ticket-monitor -n 50
sudo systemctl restart ticket-monitor.timer
```

### Playwright browser issues
```bash
cd ~/ticket-price-predictor
source venv/bin/activate
playwright install-deps chromium
```

### Out of disk space
```bash
df -h
rm ~/ticket-price-predictor/logs/monitor.log.old
```

### Connection issues
```bash
# Check security group allows SSH
aws ec2 describe-security-groups --group-names ticket-predictor-sg

# Get new public IP (if instance was stopped/started)
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ticket-predictor" \
    --query "Reservations[0].Instances[0].PublicIpAddress"
```
