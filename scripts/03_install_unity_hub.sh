#!/usr/bin/env bash
# Install Unity Hub via Unity's official apt repository (M3a.1).
#
# Unity stopped shipping a standalone AppImage; the apt repo is now the only
# supported Linux install path, which means root is unavoidable. Run as:
#
#   sudo bash scripts/03_install_unity_hub.sh
#
# After this completes, see README.md (M3a section) for the manual post-install
# steps (sign-in + Unity 2022.3 LTS install with Android Build Support module).
#
# Source: https://docs.unity.com/en-us/hub/install-hub-linux
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: must be run as root. Re-run as: sudo bash $0" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "==> installing curl"
    apt update
    apt install -y curl
fi

echo "==> adding Unity public signing key"
install -d /etc/apt/keyrings
curl -fsSL https://hub.unity3d.com/linux/keys/public \
    | gpg --dearmor -o /etc/apt/keyrings/unityhub.gpg

echo "==> adding unityhub apt source"
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/unityhub.gpg] https://hub.unity3d.com/linux/repos/deb stable main" \
    > /etc/apt/sources.list.d/unityhub.list

echo "==> apt update + install unityhub"
apt update
apt install -y unityhub

echo
echo "Unity Hub installed. Launch via 'unityhub' (or your application menu)."
echo "Then follow README.md M3a -> 'Sign in and install Unity Editor'."
