#!/bin/bash
# Create systemd service and timer for hourly data collection

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
ExecStart=/home/ubuntu/ticket-price-predictor/venv/bin/python scripts/monitor_popular.py
StandardOutput=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log
StandardError=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Create timer file (runs hourly)
sudo tee /etc/systemd/system/ticket-monitor.timer > /dev/null << 'EOF'
[Unit]
Description=Ticket Price Monitor Timer
Requires=ticket-monitor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Unit=ticket-monitor.service

[Install]
WantedBy=timers.target
EOF

# Reload systemd and enable timer
echo "Enabling and starting timer..."
sudo systemctl daemon-reload
sudo systemctl enable ticket-monitor.timer
sudo systemctl start ticket-monitor.timer

echo ""
echo "=== Service Installed ==="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ticket-monitor.timer    # Check timer status"
echo "  sudo systemctl status ticket-monitor.service  # Check service status"
echo "  sudo journalctl -u ticket-monitor -f          # View logs"
echo "  tail -f ~/ticket-price-predictor/logs/monitor.log  # View collection logs"
echo ""
echo "Timer will run every hour starting 5 minutes after boot."
