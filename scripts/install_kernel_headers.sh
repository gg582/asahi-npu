#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

DKMS_CONF="${REPO_ROOT}/dkms.conf"
if [[ ! -f "${DKMS_CONF}" ]]; then
    echo "[install-kernel-headers] dkms.conf not found at ${DKMS_CONF}." >&2
    exit 4
fi

dkms_name_from_conf=$(awk -F'"' '/^PACKAGE_NAME="/ {print $2}' "${DKMS_CONF}" | head -n1)
dkms_version_from_conf=$(awk -F'"' '/^PACKAGE_VERSION="/ {print $2}' "${DKMS_CONF}" | head -n1)

DKMS_NAME=${DKMS_NAME:-${dkms_name_from_conf:-ane}}
DKMS_VERSION=${DKMS_VERSION:-${dkms_version_from_conf:-1.0}}
DKMS_SRC_DIR="/usr/src/${DKMS_NAME}-${DKMS_VERSION}"

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
    "rsync"
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

echo "[install-kernel-headers] Preparing DKMS source tree at ${DKMS_SRC_DIR}"
if ! command -v rsync >/dev/null 2>&1; then
    echo "[install-kernel-headers] rsync is required but not available." >&2
    exit 3
fi
mkdir -p "${DKMS_SRC_DIR}"
rsync -a --delete \
    --exclude '.git/' \
    --exclude '.gitignore' \
    --exclude '.github/' \
    --exclude 'build/' \
    --exclude '*.o' \
    --exclude '*.ko' \
    --exclude '*~' \
    "${REPO_ROOT}/" "${DKMS_SRC_DIR}/"

if [[ -d "/var/lib/dkms/${DKMS_NAME}/${DKMS_VERSION}" ]]; then
    echo "[install-kernel-headers] Removing pre-existing DKMS entry for ${DKMS_NAME}/${DKMS_VERSION}"
    dkms remove -m "${DKMS_NAME}" -v "${DKMS_VERSION}" --all || true
fi

echo "[install-kernel-headers] Registering ANE module with DKMS"
dkms add -m "${DKMS_NAME}" -v "${DKMS_VERSION}" || {
    echo "[install-kernel-headers] DKMS add failed; cleaning up stale state and retrying" >&2
    dkms remove -m "${DKMS_NAME}" -v "${DKMS_VERSION}" --all || true
    dkms add -m "${DKMS_NAME}" -v "${DKMS_VERSION}"
}

echo "[install-kernel-headers] Building ANE module via DKMS"
dkms build -m "${DKMS_NAME}" -v "${DKMS_VERSION}"

echo "[install-kernel-headers] Installing ANE module via DKMS"
dkms install -m "${DKMS_NAME}" -v "${DKMS_VERSION}" --force

echo "[install-kernel-headers] DKMS installation complete. Use 'modprobe ${DKMS_NAME}' to load the module."
