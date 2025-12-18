#!/usr/bin/env bash
set -e

# Convenience wrapper: 1GPU run (same as run_srun.sh).
exec bash "$(dirname "$0")/run_srun.sh" "$@"


