#!/bin/bash
# EC2 Instance Setup Script
# Run this on the EC2 instance after SSH connection

set -e

echo "=== EC2 Setup for Ticket Price Predictor ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.11+
echo "Installing Python 3.11..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies for Playwright
echo "Installing Playwright system dependencies..."
sudo apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2

# Create project directory
echo "Creating project directory..."
mkdir -p ~/ticket-price-predictor
cd ~/ticket-price-predictor

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install \
    pydantic \
    pyarrow \
    pandas \
    lightgbm \
    scikit-learn \
    playwright \
    playwright-stealth \
    python-dotenv \
    requests

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium
playwright install-deps chromium

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw/listings logs

# Create .env file template
echo "Creating .env template..."
cat > .env << 'EOF'
# Optional API keys for popularity data
# SPOTIFY_CLIENT_ID=
# SPOTIFY_CLIENT_SECRET=
# SONGKICK_API_KEY=
# BANDSINTOWN_APP_ID=
EOF

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Copy your project files to ~/ticket-price-predictor/"
echo "2. Set up the systemd service (see ec2-service.sh)"
echo "3. Enable the timer for hourly collection"
