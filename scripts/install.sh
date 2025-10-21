#!/bin/bash

# --- Configuration ---
# Must match PACKAGE_NAME and PACKAGE_VERSION in dkms.conf
PACKAGE_NAME="asahi-npu"
PACKAGE_VERSION="0.1"
SOURCE_DIR=$(pwd)
DEST_DIR="/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}"

# Check for root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

echo "--- Starting DKMS Installation for ${PACKAGE_NAME} v${PACKAGE_VERSION} ---"

# Step 1: Copy the source files to the DKMS location
echo "1. Copying source files to ${DEST_DIR}..."
rsync -av "${SOURCE_DIR}/" "${DEST_DIR}/"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy files."
    exit 1
fi

# Step 2: Add the module source to DKMS
echo "2. Adding module source to DKMS tree..."
dkms add -m "${PACKAGE_NAME}" -v "${PACKAGE_VERSION}"

if [ $? -ne 0 ]; then
    echo "ERROR: DKMS failed to add the module. Check dkms.conf syntax."
    exit 1
fi

# Step 3: Build the module for the current kernel
echo "3. Building module for the current kernel ($(uname -r))..."
dkms build -m "${PACKAGE_NAME}" -v "${PACKAGE_VERSION}"

if [ $? -ne 0 ]; then
    echo "ERROR: DKMS build failed. Check your Makefile and kernel headers."
    exit 1
fi

# Step 4: Install the built module
echo "4. Installing module..."
dkms install -m "${PACKAGE_NAME}" -v "${PACKAGE_VERSION}"

if [ $? -ne 0 ]; then
    echo "ERROR: DKMS install failed."
    exit 1
fi

# Step 5: Verification
echo ""
echo "--- Installation Complete ---"
echo "Current DKMS status:"
dkms status
echo ""
echo "To load the module immediately (replace 'ane_drv' with the module name from dkms.conf):"
echo "  modprobe ane_drv"
