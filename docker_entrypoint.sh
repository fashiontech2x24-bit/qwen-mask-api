#!/bin/bash
# Docker entrypoint script

# If first arg is 'python' or 'python3', replace with python3.10
if [ "$1" = "python" ] || [ "$1" = "python3" ]; then
    shift
    exec python3.10 "$@"
else
    # Otherwise run as-is
    exec "$@"
fi

