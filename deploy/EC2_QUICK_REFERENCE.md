# AWS EC2 Deployment - Quick Reference

## Instance Details
- **Instance ID:** i-0e0fc71a207fedc21
- **Public IP:** 3.85.167.225
- **Instance Type:** t3.micro (free tier)
- **Region:** us-east-1
- **SSH Key:** ~/.ssh/ticket-predictor-key.pem

## Status: ✅ Running

**Hourly data collection is active!**
- Timer runs every hour
- Successfully tested: Collected 1,290 listings from 15 events
- Next collection: Check with `ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@3.85.167.225 "sudo systemctl list-timers"`

## Common Commands

### SSH Connection
```bash
ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@3.85.167.225
```

### Check Service Status
```bash
# On EC2
sudo systemctl status ticket-monitor.timer
sudo systemctl status ticket-monitor.service
sudo journalctl -u ticket-monitor -f
tail -f ~/ticket-price-predictor/logs/monitor.log
```

### Sync Data to Local
```bash
# From local machine
cd /Users/heather/ticket-price-predictor
./deploy/sync-from-ec2.sh 3.85.167.225
```

### Stop/Start Collection
```bash
# On EC2
sudo systemctl stop ticket-monitor.timer
sudo systemctl start ticket-monitor.timer
```

### Manual Test Run
```bash
# On EC2
cd ~/ticket-price-predictor
source venv/bin/activate
python monitor_popular.py
```

## AWS Management

### Stop Instance (Saves Money)
```bash
aws ec2 stop-instances --instance-ids i-0e0fc71a207fedc21
```

### Start Instance
```bash
aws ec2 start-instances --instance-ids i-0e0fc71a207fedc21
# Note: Public IP will change, update sync script
```

### Check Instance Status
```bash
aws ec2 describe-instances --instance-ids i-0e0fc71a207fedc21 --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text
```

### View Billing
https://console.aws.amazon.com/billing/home#/freetier

## Cost Optimization

**Free Tier Limits:**
- ✅ 750 hours/month of t3.micro (covers 24/7)
- ✅ 30 GB EBS storage
- ✅ 15 GB bandwidth outbound

**Current Usage:**
- Instance: Running 24/7 (within free tier)
- Storage: ~5 GB (well within limit)
- Bandwidth: ~50 GB/month (may exceed by ~$3/month)

**Estimated Cost:** $0-4/month depending on bandwidth

## Troubleshooting

### Service Won't Start
```bash
sudo journalctl -u ticket-monitor -n 50
```

### Missing Dependencies
```bash
cd ~/ticket-price-predictor
source venv/bin/activate
pip install httpx pydantic-settings selectolax spotipy
```

### Out of Disk Space
```bash
df -h
# Clean logs if needed
rm ~/ticket-price-predictor/logs/*.old
```

## Next Steps

1. **Set up daily sync** (optional):
   ```bash
   # Add to local crontab
   0 8 * * * cd /Users/heather/ticket-price-predictor && ./deploy/sync-from-ec2.sh 3.85.167.225 >> logs/sync.log 2>&1
   ```

2. **Monitor costs** at AWS Billing Console

3. **Train models locally** with growing dataset

## Important Notes

- Keep the instance running 24/7 to stay within free tier
- Public IP changes if you stop/start the instance
- SSH key is stored at `~/.ssh/ticket-predictor-key.pem`
- Security group allows SSH from anywhere (port 22)
- Timer runs hourly, starting 5 minutes after boot
