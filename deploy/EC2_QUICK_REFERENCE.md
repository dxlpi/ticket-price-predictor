# AWS EC2 Deployment - Quick Reference

## Instance Details
- **Instance ID:** i-0e0fc71a207fedc21
- **Public IP:** 3.85.167.225
- **Instance Type:** t3.micro (free tier)
- **Region:** us-east-1
- **SSH Key:** ~/.ssh/ticket-predictor-key.pem

## Status: Running

**Hourly data collection is active.**
- Timer runs every hour
- Data sources: VividSeats scraper + YouTube Music + Last.fm popularity
- Script: `scripts/monitor_popular.py`

## Common Commands

### SSH Connection
```bash
ssh -i ~/.ssh/ticket-predictor-key.pem ubuntu@3.85.167.225
```

### Deploy Code Updates
```bash
# From local machine (project root)
./deploy/deploy-to-ec2.sh
```
This rsyncs source code, transfers `.env`, installs packages, and restarts the systemd timer.

### Sync Data to Local
```bash
# From local machine (project root)
./deploy/sync-from-ec2.sh
# Or with explicit IP:
./deploy/sync-from-ec2.sh 3.85.167.225
```

### Check Service Status (on EC2)
```bash
sudo systemctl status ticket-monitor.timer
sudo systemctl status ticket-monitor.service
sudo journalctl -u ticket-monitor -f
tail -f ~/ticket-price-predictor/logs/monitor.log
```

### Stop/Start Collection (on EC2)
```bash
sudo systemctl stop ticket-monitor.timer
sudo systemctl start ticket-monitor.timer
```

### Manual Test Run (on EC2)
```bash
cd ~/ticket-price-predictor
source venv/bin/activate
python scripts/monitor_popular.py
```

## Automated Daily Sync

Add to local crontab (`crontab -e`):
```bash
0 8 * * * cd /Users/heather/ticket-price-predictor && ./deploy/sync-from-ec2.sh >> logs/sync.log 2>&1
```

## Environment Variables (.env)
```
TICKETMASTER_API_KEY=...
LASTFM_API_KEY=...
# YouTube Music uses ytmusicapi (no API key needed)
```

## AWS Management

### Stop Instance (Saves Money)
```bash
aws ec2 stop-instances --instance-ids i-0e0fc71a207fedc21
```

### Start Instance
```bash
aws ec2 start-instances --instance-ids i-0e0fc71a207fedc21
# Note: Public IP may change — update deploy scripts if so
```

### Check Instance Status
```bash
aws ec2 describe-instances --instance-ids i-0e0fc71a207fedc21 \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text
```

## Cost Optimization

**Free Tier Limits:**
- 750 hours/month of t3.micro (covers 24/7)
- 30 GB EBS storage
- 15 GB bandwidth outbound

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
pip install ytmusicapi httpx pydantic python-dotenv
```

### Out of Disk Space
```bash
df -h
# Clean logs if needed
rm ~/ticket-price-predictor/logs/*.old
```
