#!/bin/sh
set -eu

if [ -n "${HALLU_GATEWAY_API_KEY_FILE:-}" ]; then
    HALLU_GATEWAY_API_KEY="$(cat "$HALLU_GATEWAY_API_KEY_FILE")"
    export HALLU_GATEWAY_API_KEY
fi

exec python -m det_rep "$@"
