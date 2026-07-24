#!/bin/bash

# Compatibility alias for the Ref-NeRF batch script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_refnerf.sh" "$@"
