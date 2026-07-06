#!/usr/bin/env bash
set -euo pipefail

# Install the external reconstruction CLIs without adding their dependencies to
# ComfyUI's Python environment. Ubuntu's COLMAP package is CPU-only; Tinode's
# COLMAP node therefore defaults to CPU SIFT.

VENV_PATH="${TINODE_RECON_VENV:-/workspace/nerfstudio-venv}"
COLMAP_ONLY=0

while (($#)); do
	case "$1" in
		--colmap-only) COLMAP_ONLY=1 ;;
		--venv)
			shift
			VENV_PATH="${1:?--venv requires a path}"
			;;
		-h|--help)
			echo "Usage: $0 [--colmap-only] [--venv PATH]"
			exit 0
			;;
		*) echo "Unknown argument: $1" >&2; exit 2 ;;
	esac
	shift
done

if ! command -v apt-get >/dev/null 2>&1; then
	echo "This installer currently supports Debian/Ubuntu (apt-get)." >&2
	exit 1
fi

if [[ $(id -u) -eq 0 ]]; then
	APT=(apt-get)
elif command -v sudo >/dev/null 2>&1; then
	APT=(sudo apt-get)
else
	echo "Root or sudo is required to install COLMAP." >&2
	exit 1
fi

"${APT[@]}" update
"${APT[@]}" install -y colmap python3-venv

echo "COLMAP: $(command -v colmap)"
colmap -h >/dev/null

if [[ $COLMAP_ONLY -eq 1 ]]; then
	exit 0
fi

python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install nerfstudio

echo
echo "Use these values in Tinode:"
echo "  process_command: $VENV_PATH/bin/ns-process-data"
echo "  train_command:   $VENV_PATH/bin/ns-train"
echo "  export_command:  $VENV_PATH/bin/ns-export"
