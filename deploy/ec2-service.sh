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
EnvironmentFile=/home/ubuntu/ticket-price-predictor/.env
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

# Create urgent service (near-term events only, every 30 min)
sudo tee /etc/systemd/system/ticket-monitor-urgent.service > /dev/null << 'EOF'
[Unit]
Description=Ticket Price Monitor (Urgent - near-term events)
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/ticket-price-predictor
Environment="PATH=/home/ubuntu/ticket-price-predictor/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ubuntu/ticket-price-predictor/.env
ExecStart=/home/ubuntu/ticket-price-predictor/venv/bin/python scripts/monitor_popular.py --urgent
StandardOutput=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log
StandardError=append:/home/ubuntu/ticket-price-predictor/logs/monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Create urgent timer (runs every 30 minutes)
sudo tee /etc/systemd/system/ticket-monitor-urgent.timer > /dev/null << 'EOF'
[Unit]
Description=Ticket Price Monitor Timer (Urgent)
Requires=ticket-monitor-urgent.service

[Timer]
OnBootSec=15min
OnUnitActiveSec=30min
Unit=ticket-monitor-urgent.service

[Install]
WantedBy=timers.target
EOF

# Reload systemd and enable timers
echo "Enabling and starting timers..."
sudo systemctl daemon-reload
sudo systemctl enable ticket-monitor.timer
sudo systemctl start ticket-monitor.timer
sudo systemctl enable ticket-monitor-urgent.timer
sudo systemctl start ticket-monitor-urgent.timer

echo ""
echo "=== Services Installed ==="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ticket-monitor.timer         # Full collection (hourly)"
echo "  sudo systemctl status ticket-monitor-urgent.timer  # Urgent collection (30 min)"
echo "  sudo systemctl status ticket-monitor.service       # Check service status"
echo "  sudo journalctl -u ticket-monitor -f               # View logs"
echo "  tail -f ~/ticket-price-predictor/logs/monitor.log  # View collection logs"
echo ""
echo "Full collection runs hourly. Urgent (≤14 day events) runs every 30 minutes."
