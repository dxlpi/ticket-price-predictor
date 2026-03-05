#!/bin/bash
# Create systemd service and timer for data collection (every 8 hours)

set -e

echo "=== Creating systemd service and timer ==="

# Create service file
sudo tee /etc/systemd/system/ticket-monitor.service > /dev/null << 'EOF'
[Unit]
Description=Ticket Price Monitor
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/ticket-price-predictor
Environment="PATH=/home/ubuntu/ticket-price-predictor/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ubuntu/ticket-price-predictor/.env
ExecStart=/home/ubuntu/ticket-price-predictor/venv/bin/python scripts/monitor_popular.py
StandardOutput=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log
StandardError=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Create timer file (runs every 8 hours)
sudo tee /etc/systemd/system/ticket-monitor.timer > /dev/null << 'EOF'
[Unit]
Description=Ticket Price Monitor Timer
Requires=ticket-monitor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=8h
Unit=ticket-monitor.service

[Install]
WantedBy=timers.target
EOF

# Reload systemd and enable timer
echo "Enabling and starting timer..."
sudo systemctl daemon-reload

# Stop and disable old urgent timer if it exists
sudo systemctl stop ticket-monitor-urgent.timer 2>/dev/null || true
sudo systemctl disable ticket-monitor-urgent.timer 2>/dev/null || true
sudo rm -f /etc/systemd/system/ticket-monitor-urgent.service
sudo rm -f /etc/systemd/system/ticket-monitor-urgent.timer

sudo systemctl enable ticket-monitor.timer
sudo systemctl start ticket-monitor.timer

echo ""
echo "=== Service Installed ==="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ticket-monitor.timer    # Collection timer (every 8h)"
echo "  sudo systemctl status ticket-monitor.service  # Check service status"
echo "  sudo journalctl -u ticket-monitor -f          # View logs"
echo "  tail -f ~/ticket-price-predictor/logs/monitor.log  # View collection logs"
echo ""
echo "Collection runs every 8 hours (~3 times per day)."
