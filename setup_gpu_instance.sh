#!/bin/bash

# Complete NVIDIA driver installation for plain Ubuntu 24.04
# Includes Docker (API 1.52+), Docker Compose, and Git
# Supports post-reboot configuration

set -e

LOGFILE="/var/log/nvidia-install.log"
exec > >(sudo tee -a $LOGFILE)
exec 2>&1

echo "=========================================="
echo "NVIDIA Driver Installation - $(date)"
echo "=========================================="

# Check if this is post-reboot execution
if [ -f "/var/log/.nvidia-post-reboot" ]; then
    echo "Running post-reboot configuration..."
    
    # Enable NVIDIA persistence mode
    sudo nvidia-smi -pm 1 || true
    
    # Verify GPU is accessible
    echo "Verifying GPU access..."
    sudo nvidia-smi
    
    # Test Docker GPU access
    echo "Testing Docker GPU access..."
    sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi || {
        echo "ERROR: Docker cannot access GPU. Reconfiguring..."
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        sleep 5
        sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
    }
    
    # ============================================================
    # START CUSTOM POST-REBOOT COMMANDS
    # ============================================================
    
    echo "Setting up environment and tokens..."
    
    # Export tokens for this session
    # IMPORTANT: Replace these placeholder values with your actual tokens before running!
    export GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_GITHUB_TOKEN_HERE}"
    export NEWSAPI_API_KEY="${NEWSAPI_API_KEY:-YOUR_NEWSAPI_KEY_HERE}"
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-YOUR_HF_TOKEN_HERE}"
    export OPEN_AI_API_KEY="${OPEN_AI_API_KEY:-YOUR_OPENAI_KEY_HERE}"
    
    # Validate that tokens are set
    if [[ "$GITHUB_TOKEN" == "YOUR_GITHUB_TOKEN_HERE" ]] || \
       [[ "$NEWSAPI_API_KEY" == "YOUR_NEWSAPI_KEY_HERE" ]] || \
       [[ "$HUGGING_FACE_HUB_TOKEN" == "YOUR_HF_TOKEN_HERE" ]] || \
       [[ "$OPEN_AI_API_KEY" == "YOUR_OPENAI_KEY_HERE" ]]; then
        echo "ERROR: Please set your API tokens before running this script!"
        echo "Edit this script and replace the placeholder values with your actual tokens."
        exit 1
    fi
    
    # Login to GitHub Container Registry with sudo
    echo $GITHUB_TOKEN | sudo docker login ghcr.io -u michaldobiezynski --password-stdin
    
    # Create necessary directories
    echo "Creating directories..."
    sudo mkdir -p /tmp/models
    sudo mkdir -p ./volumes/{hf-cache,cargo-registry-tts,cargo-registry-stt,tts-target,stt-target,tts-logs,stt-logs,uv-cache}
    
    # Clone repository (or pull if exists)
    echo "Cloning repository..."
    if [ -d "unmute" ]; then
        echo "Repository already exists, pulling latest..."
        cd unmute
        sudo git pull
        
        # Clean up any existing containers and volumes to ensure fresh build
        echo "Cleaning up existing Docker resources..."
        sudo docker compose down -v || true
        
        # Remove the Cargo build cache volumes that might contain failed builds
        echo "Removing stale Cargo build caches..."
        sudo rm -rf ./volumes/tts-target/* || true
        sudo rm -rf ./volumes/stt-target/* || true
        sudo rm -rf ./volumes/cargo-registry-tts/* || true
        sudo rm -rf ./volumes/cargo-registry-stt/* || true
    else
        sudo git clone https://github.com/michaldobiezynski/unmute.git
        cd unmute
    fi
    
    # Create .env file with tokens for docker compose
    # We add CUDA_COMPUTE_CAP=80 here to fix the NVML initialization error during build
    sudo tee .env > /dev/null << ENVEOF
NEWSAPI_API_KEY=${NEWSAPI_API_KEY}
HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
OPEN_AI_API_KEY=${OPEN_AI_API_KEY}
GITHUB_TOKEN=${GITHUB_TOKEN}
CUDA_COMPUTE_CAP=80
ENVEOF

    # Create docker-compose.override.yml to inject CUDA_COMPUTE_CAP into services
    # We hardcode the value (80) instead of using variable substitution
    # to avoid issues with sudo and environment variable expansion
    sudo tee docker-compose.override.yml > /dev/null << 'OVERRIDEOF'
services:
  tts:
    environment:
      - CUDA_COMPUTE_CAP=80
  stt:
    environment:
      - CUDA_COMPUTE_CAP=80
OVERRIDEOF
    
    # Start services with docker compose (detached mode with sudo)
    echo "Starting services with docker compose..."
    sudo -E CUDA_COMPUTE_CAP=80 docker compose up --build -d
    
    echo "Services started successfully!"
    sudo docker ps -a
    
    # Show logs for debugging
    echo "Checking service logs..."
    sudo docker compose logs tts stt | tail -50
    
    # ============================================================
    # END CUSTOM POST-REBOOT COMMANDS
    # ============================================================
    
    # Schedule shutdown in 1 hour
    sudo shutdown -h +60 "System will shut down in 60 minutes. Please save your work."
    
    echo "Post-reboot configuration complete!"
    
    # Clean up - remove the service and marker
    sudo rm -f /var/log/.nvidia-post-reboot
    sudo rm -f /etc/systemd/system/nvidia-post-install.service
    sudo rm -f /home/ubuntu/install-gpu.sh
    sudo systemctl daemon-reload
    
    exit 0
fi

# Check if already installed
if [ -f "/var/log/.nvidia-installed" ]; then
    echo "Drivers already installed, enabling persistence mode..."
    sudo nvidia-smi -pm 1 || true
    exit 0
fi

# ============================================================
# SAVE THIS SCRIPT TO DISK (for post-reboot execution)
# ============================================================
if [ ! -f "/home/ubuntu/install-gpu.sh" ]; then
    echo "Saving script for post-reboot execution..."
    sudo cp "$0" /home/ubuntu/install-gpu.sh
    sudo chmod +x /home/ubuntu/install-gpu.sh
fi

# ============================================================
# Clean up any broken installations first
echo "Cleaning up any broken installations..."
sudo dpkg --configure -a || true
sudo apt-get install -f -y || true
sudo apt-get remove --purge -y nvidia-* || true
sudo apt-get autoremove -y || true
sudo apt-get autoclean -y || true

# Update and install prerequisites
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    ubuntu-drivers-common \
    dkms \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Install Git
echo "Installing Git..."
sudo apt-get install -y git

# Install Docker (latest version for API 1.52 support)
echo "Installing Docker..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Set DOCKER_API_VERSION environment variable system-wide
echo "Setting DOCKER_API_VERSION=1.52..."
echo "export DOCKER_API_VERSION=1.52" | sudo tee -a /etc/environment
echo "export DOCKER_API_VERSION=1.52" | sudo tee /etc/profile.d/docker-api.sh
sudo chmod +x /etc/profile.d/docker-api.sh

# Verify Docker version
DOCKER_VERSION=$(sudo docker version --format '{{.Server.Version}}')
echo "Installed Docker version: $DOCKER_VERSION"

# Install NVIDIA driver 570-server
echo "Installing NVIDIA driver 570-server..."
echo "Current kernel: $(uname -r)"
sudo apt-get install -y \
    nvidia-driver-570-server \
    nvidia-utils-570-server \
    nvidia-dkms-570-server

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker for NVIDIA - this is critical!
echo "Configuring NVIDIA Container Runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Verify the configuration was applied
echo "Verifying Docker daemon configuration..."
cat /etc/docker/daemon.json

# Restart Docker to apply NVIDIA runtime configuration
echo "Restarting Docker..."
sudo systemctl restart docker

# Wait for Docker to be ready
sleep 5

# Mark as installed BEFORE reboot
sudo touch /var/log/.nvidia-installed

# Create post-reboot service to run this script again
echo "Creating post-reboot service..."
sudo touch /var/log/.nvidia-post-reboot

cat << 'SERVICEEOF' | sudo tee /etc/systemd/system/nvidia-post-install.service > /dev/null
[Unit]
Description=NVIDIA Post-Install Configuration
After=network.target docker.service

[Service]
Type=oneshot
ExecStart=/home/ubuntu/install-gpu.sh
RemainAfterExit=no
WorkingDirectory=/home/ubuntu

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl enable nvidia-post-install.service

# Create MOTD
cat << 'EOF' | sudo tee /etc/motd > /dev/null
========================================
✓ NVIDIA GPU Instance Ready
========================================
NVIDIA drivers installed and configured.
Docker & Docker Compose installed.
Git installed.
DOCKER_API_VERSION=1.52 configured.
Run 'nvidia-smi' to check GPU status.
Run 'sudo docker --version' to verify Docker.
Run 'sudo docker compose version' to verify Compose.

Your services have been automatically started!
Check logs: cd ~/unmute && sudo docker compose logs -f
EOF

echo "Installation complete, rebooting to load drivers..."
echo "=========================================="

# Reboot to load kernel modules
sudo shutdown -r now

