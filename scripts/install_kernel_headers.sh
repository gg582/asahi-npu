#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo -- "$0" "$@"
    else
        echo "[install-kernel-headers] Please run this script as root or install sudo." >&2
        exit 1
    fi
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "[install-kernel-headers] apt-get is required on Debian/Ubuntu systems." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

KERNEL_RELEASE=${KERNEL_RELEASE:-$(uname -r)}
PACKAGE_PREFIX=${PACKAGE_PREFIX:-linux-headers}

packages=(
    "${PACKAGE_PREFIX}-${KERNEL_RELEASE}"
    "build-essential"
    "dkms"
)

echo "[install-kernel-headers] Updating package lists..."
apt-get update

echo "[install-kernel-headers] Installing packages: ${packages[*]}"
apt-get install -y --no-install-recommends "${packages[@]}"

echo "[install-kernel-headers] Verifying header directory for ${KERNEL_RELEASE}"
if [[ ! -d "/lib/modules/${KERNEL_RELEASE}/build" ]]; then
    echo "[install-kernel-headers] Warning: /lib/modules/${KERNEL_RELEASE}/build was not created." >&2
    exit 2
fi

echo "[install-kernel-headers] Kernel headers and DKMS dependencies installed."
