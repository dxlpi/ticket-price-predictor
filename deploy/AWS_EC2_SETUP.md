# AWS EC2 Free Tier Setup Guide

Complete guide to deploy the ticket price predictor on AWS EC2 free tier.

## Prerequisites

- AWS account (with free tier eligibility)
- AWS CLI installed and configured locally
- SSH key pair for EC2 access

## Step 1: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# You'll be prompted for:
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

# Launch t2.micro instance (free tier)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t2.micro \
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
   - **Instance type:** t2.micro (Free tier eligible)
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
# Replace [EC2_PUBLIC_IP] with your instance IP
scp -i ~/.ssh/ticket-predictor-key.pem \
    deploy/ec2-setup.sh \
    deploy/ec2-service.sh \
    ubuntu@[EC2_PUBLIC_IP]:~/
```

On the **EC2 instance** (via SSH):

```bash
# Run setup script
chmod +x ec2-setup.sh ec2-service.sh
./ec2-setup.sh

# This will:
# - Install Python 3.11
# - Install Playwright and dependencies
# - Create virtual environment
# - Set up project directories
```

## Step 5: Copy Project Files

On your **local machine**:

```bash
# Copy project code to EC2
rsync -avz --progress \
    -e "ssh -i ~/.ssh/ticket-predictor-key.pem" \
    --exclude 'venv/' \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '.omc/' \
    --exclude 'logs/' \
    src/ scripts/ pyproject.toml \
    ubuntu@[EC2_PUBLIC_IP]:~/ticket-price-predictor/
```

## Step 6: Install Service

On the **EC2 instance**:

```bash
# Install systemd service
./ec2-service.sh

# Check timer status
sudo systemctl status ticket-monitor.timer

# View logs
tail -f ~/ticket-price-predictor/logs/monitor.log
```

## Step 7: Sync Data Back to Local

On your **local machine** (run daily or as needed):

```bash
# Make sync script executable
chmod +x deploy/sync-from-ec2.sh

# Sync data from EC2
./deploy/sync-from-ec2.sh [EC2_PUBLIC_IP]
```

## Management Commands

### On EC2 Instance

```bash
# Check timer status
sudo systemctl status ticket-monitor.timer

# Check service status
sudo systemctl status ticket-monitor.service

# View service logs
sudo journalctl -u ticket-monitor -f

# View collection logs
tail -f ~/ticket-price-predictor/logs/monitor.log

# Manually trigger collection
cd ~/ticket-price-predictor
source venv/bin/activate
python scripts/monitor_popular.py

# Stop timer
sudo systemctl stop ticket-monitor.timer

# Start timer
sudo systemctl start ticket-monitor.timer
```

### On Local Machine

```bash
# Sync data from EC2
./deploy/sync-from-ec2.sh [EC2_PUBLIC_IP]

# SSH into EC2
ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@[EC2_PUBLIC_IP]

# Check EC2 instance status
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ticket-predictor" \
    --query "Reservations[0].Instances[0].State.Name"

# Stop EC2 instance (saves costs when not needed)
aws ec2 stop-instances --instance-ids [INSTANCE_ID]

# Start EC2 instance
aws ec2 start-instances --instance-ids [INSTANCE_ID]
```

## Cost Optimization

**Free Tier Limits:**
- 750 hours/month of t2.micro (24/7 for one instance)
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
# Clean old logs
cd ~/ticket-price-predictor
rm logs/monitor.log.old
```

### Connection issues
```bash
# Check security group allows SSH
aws ec2 describe-security-groups \
    --group-names ticket-predictor-sg

# Get new public IP (if instance was stopped/started)
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ticket-predictor" \
    --query "Reservations[0].Instances[0].PublicIpAddress"
```
