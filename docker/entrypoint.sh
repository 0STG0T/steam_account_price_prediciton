#!/bin/bash

# Make CLI script executable
chmod +x /app/docker/cli.py

# Run CLI with provided arguments
/app/docker/cli.py "$@"
