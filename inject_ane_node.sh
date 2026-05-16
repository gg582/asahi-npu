#!/usr/bin/env bash
# inject_ane_node.sh - Script to inject Apple Neural Engine (ANE) nodes into the Asahi Linux Device Tree (DTB).
# This script decompiles the current DTB, runs a Python patcher to insert the required PMGR and SOC nodes,
# recompiles it, and backs up/overwrites the system DTB in /boot.
# Run with sudo.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)."
    exit 1
fi

KERNEL_VER=$(uname -r)
DTB_DIR="/boot/dtb-${KERNEL_VER}/apple"
DTB_FILE="t8103-j293.dtb" # Target for M1 (MacBook Pro 13" 2020 / Mac mini / Air depending on exact model, usually j293/j274)
DTB_PATH="${DTB_DIR}/${DTB_FILE}"

# Check if target DTB exists
if [ ! -f "${DTB_PATH}" ]; then
    # Fallback to finding the currently used compatible string if j293 isn't the primary
    COMPAT=$(tr -d '\0' < /proc/device-tree/compatible | sed 's/apple,.*//')
    # Try to find a matching dtb based on the board
    BOARD_NAME=$(tr '\0' '\n' < /proc/device-tree/compatible | head -n 1 | cut -d',' -f2)
    DTB_FILE="t8103-${BOARD_NAME}.dtb"
    DTB_PATH="${DTB_DIR}/${DTB_FILE}"
    if [ ! -f "${DTB_PATH}" ]; then
         echo "Error: Could not find DTB file at ${DTB_PATH}"
         echo "Available DTBs in ${DTB_DIR}:"
         ls -l "${DTB_DIR}"
         exit 1
    fi
fi

echo "Target DTB: ${DTB_PATH}"

# Install dtc if missing
if ! command -v dtc &> /dev/null; then
    echo "Installing Device Tree Compiler (dtc)..."
    dnf install -y dtc
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TMP_DIR"' EXIT

DTS_FILE="${TMP_DIR}/current.dts"
PATCHED_DTS_FILE="${TMP_DIR}/patched.dts"
NEW_DTB_FILE="${TMP_DIR}/new.dtb"

echo "Decompiling DTB to DTS..."
dtc -I dtb -O dts "${DTB_PATH}" > "${DTS_FILE}" 2>/dev/null || true

echo "Patching DTS via Python..."
cat << 'EOF' > "${TMP_DIR}/patch.py"
import sys

def patch_dts(dts_path, out_path):
    with open(dts_path, 'r') as f:
        lines = f.readlines()

    # Check if already patched
    if any('apple,t8103-ane' in line for line in lines):
        print("DTB already contains ANE nodes. Skipping patch.")
        sys.exit(0)

    # 1. Find AIC phandle
    aic_phandle = None
    for i, line in enumerate(lines):
        if 'compatible = "apple,t8103-aic"' in line:
            for j in range(i, i+10):
                if 'phandle = <' in lines[j]:
                    aic_phandle = lines[j].split('<')[1].split('>')[0]
                    break
            break
    if not aic_phandle:
        print("Error: Could not find aic phandle")
        sys.exit(1)
        
    # 2. Find ane_sys phandle and add always-on
    ane_sys_phandle = None
    for i, line in enumerate(lines):
        if 'label = "ane_sys";' in line:
            for j in range(i, i+5):
                if 'phandle = <' in lines[j]:
                    ane_sys_phandle = lines[j].split('<')[1].split('>')[0]
                    break
            lines.insert(i+1, '\t\t\t\tapple,always-on;\n')
            break
    if not ane_sys_phandle:
        print("Error: Could not find ane_sys phandle")
        sys.exit(1)

    # 3. Find ane_sys_cpu and add a phandle if it doesn't have one
    ane_sys_cpu_idx = -1
    ane_sys_cpu_phandle = "<0x1000>"
    for i, line in enumerate(lines):
        if 'label = "ane_sys_cpu";' in line:
            ane_sys_cpu_idx = i
            # check if it already has a phandle
            for j in range(i, i+5):
                if 'phandle = <' in lines[j]:
                    ane_sys_cpu_phandle = "<" + lines[j].split('<')[1].split('>')[0] + ">"
                    break
            else:
                 lines.insert(i+2, '\t\t\t\tphandle = <0x1000>;\n')
            break
    if ane_sys_cpu_idx == -1:
        print("Error: Could not find ane_sys_cpu")
        sys.exit(1)

    # 4. Insert PMGR nodes after ane_sys_cpu
    insert_pmgr_idx = -1
    for i in range(ane_sys_cpu_idx, len(lines)):
        if lines[i].strip() == '};':
            insert_pmgr_idx = i + 1
            break

    pmgr_nodes = f"""
\t\t\tpower-controller@c008 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc008 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_base";
\t\t\t\tpower-domains = {ane_sys_cpu_phandle};
\t\t\t\tphandle = <0x1001>;
\t\t\t}};
\t\t\tpower-controller@c010 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc010 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_set1";
\t\t\t\tpower-domains = <0x1001>;
\t\t\t\tphandle = <0x1002>;
\t\t\t}};
\t\t\tpower-controller@c018 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc018 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_set2";
\t\t\t\tpower-domains = <0x1001>;
\t\t\t\tphandle = <0x1003>;
\t\t\t}};
\t\t\tpower-controller@c020 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc020 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_set3";
\t\t\t\tpower-domains = <0x1001>;
\t\t\t\tphandle = <0x1004>;
\t\t\t}};
\t\t\tpower-controller@c028 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc028 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_set4";
\t\t\t\tpower-domains = <0x1001>;
\t\t\t\tphandle = <0x1005>;
\t\t\t}};
\t\t\tpower-controller@c030 {{
\t\t\t\tcompatible = "apple,t8103-pmgr-pwrstate", "apple,pmgr-pwrstate";
\t\t\t\treg = <0xc030 0x04>;
\t\t\t\t#power-domain-cells = <0x00>;
\t\t\t\t#reset-cells = <0x00>;
\t\t\t\tlabel = "ane_set5";
\t\t\t\tpower-domains = <0x1001>;
\t\t\t\tphandle = <0x1006>;
\t\t\t}};
"""
    lines.insert(insert_pmgr_idx, pmgr_nodes)

    # 5. Insert DART and ANE into /soc
    soc_end_idx = -1
    for i, line in enumerate(reversed(lines)):
        if line.strip() == '};' and lines[len(lines)-i-2].strip() == '};':
            soc_end_idx = len(lines) - i - 1
            break
            
    soc_nodes = f"""
\t\tiommu@26b800000 {{
\t\t\tcompatible = "apple,t8103-dart";
\t\t\treg = <0x02 0x6b800000 0x00 0x4000>;
\t\t\t#iommu-cells = <0x01>;
\t\t\tinterrupt-parent = <{aic_phandle}>;
\t\t\tinterrupts = <0x01 417 0x04>;
\t\t\tpower-domains = <{ane_sys_phandle}>;
\t\t\tphandle = <0x1007>;
\t\t}};

\t\tane@26bc04000 {{
\t\t\tcompatible = "apple,t8103-ane";
\t\t\tiommus = <0x1007 0x00>;
\t\t\treg-names = "engine", "dart0", "dart1", "dart2";
\t\t\treg = <0x02 0x6bc04000 0x00 0x24000>,
\t\t\t      <0x02 0x6b800000 0x00 0x4000>,
\t\t\t      <0x02 0x6b810000 0x00 0x4000>,
\t\t\t      <0x02 0x6b820000 0x00 0x4000>;
\t\t\tinterrupt-parent = <{aic_phandle}>;
\t\t\tinterrupt-names = "ane", "dart";
\t\t\tinterrupts = <0x01 416 0x04>, <0x01 417 0x04>;
\t\t\tpower-domains = <0x1002>, <0x1003>, <0x1004>, <0x1005>, <0x1006>;
\t\t}};
"""
    lines.insert(soc_end_idx - 1, soc_nodes)

    with open(out_path, 'w') as f:
        f.writelines(lines)
    print("DTS patched successfully.")

if __name__ == "__main__":
    patch_dts(sys.argv[1], sys.argv[2])
EOF

python3 "${TMP_DIR}/patch.py" "${DTS_FILE}" "${PATCHED_DTS_FILE}"

if grep -q "already contains ANE nodes" <<< "$(python3 "${TMP_DIR}/patch.py" "${DTS_FILE}" "${PATCHED_DTS_FILE}")"; then
    echo "No changes made."
    exit 0
fi

echo "Compiling patched DTS back to DTB..."
dtc -I dts -O dtb "${PATCHED_DTS_FILE}" > "${NEW_DTB_FILE}" 2>/dev/null || true

if [ ! -s "${NEW_DTB_FILE}" ]; then
    echo "Error: Failed to compile new DTB."
    exit 1
fi

echo "Backing up original DTB to ${DTB_PATH}.bak"
cp "${DTB_PATH}" "${DTB_PATH}.bak"

echo "Overwriting system DTB..."
cp "${NEW_DTB_FILE}" "${DTB_PATH}"

echo "Done! The ANE device tree nodes have been injected."
echo "Please reboot your system for the changes to take effect."
