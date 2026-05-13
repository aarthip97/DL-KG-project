#!/bin/bash

# ==============================================================================
# GraphDB System-Wide Setup for Linux
# Installs to ~/graphdb-server to act as a permanent, multi-project database
# ==============================================================================

GRAPHDB_VERSION="11.3.3"
INSTALL_DIR="$HOME/graphdb-server"
ZIP_FILE="/tmp/graphdb-${GRAPHDB_VERSION}-dist.zip"
DOWNLOAD_URL="https://download.ontotext.com/owlim/0521929f-94ab-4ac0-adce-84fc0426b69e/graphdb-11.3.3-dist.zip?_gl=1*15600kb*_ga*MTE2MzA3NTI2OS4xNzc4MTMyOTEw*_ga_HGSKWBWCRK*czE3Nzg2OTk1ODMkbzMkZzEkdDE3Nzg3MDAxMTYkajU5JGwwJGgw"

echo "=== Starting System-Wide GraphDB Setup ==="

# 1. Download and Extract to the Home Directory
if [ ! -d "$INSTALL_DIR" ]; then
    echo "[*] Downloading GraphDB $GRAPHDB_VERSION to /tmp..."
    wget -O "$ZIP_FILE" "$DOWNLOAD_URL"
    
    echo "[*] Extracting to $HOME..."
    unzip -q "$ZIP_FILE" -d "$HOME"
    
    # Rename the extracted folder to a clean, generic name
    mv "$HOME/graphdb-${GRAPHDB_VERSION}" "$INSTALL_DIR"
    rm "$ZIP_FILE"
else
    echo "[*] GraphDB is already installed at $INSTALL_DIR. Skipping download."
fi

# 2. Inject License File (if it exists in the Git repo where the script is run)
if [ -f "../graphdb.license" ]; then
    echo "[*] Found graphdb.license. Injecting into system config..."
    cp ../graphdb.license "${INSTALL_DIR}/conf/"
else
    echo "[!] No graphdb.license file found. Running in default mode."
fi

# 3. Setup Systemd Service
echo "[*] Configuring systemd background service (requires sudo)..."
CURRENT_USER=$(whoami)

SERVICE_CONTENT="[Unit]
Description=Ontotext GraphDB Multi-Project Server
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/bin/graphdb
Restart=on-failure
SuccessExitStatus=143

[Install]
WantedBy=multi-user.target"

echo "$SERVICE_CONTENT" > /tmp/graphdb.service
sudo mv /tmp/graphdb.service /etc/systemd/system/graphdb.service

# 4. Enable and Start the Daemon
sudo systemctl daemon-reload
sudo systemctl enable graphdb
sudo systemctl restart graphdb

echo "=== System Installation Complete! ==="
echo "GraphDB is permanently installed at: $INSTALL_DIR"
echo "Check status: sudo systemctl status graphdb"
echo "Web UI:       http://localhost:7200"