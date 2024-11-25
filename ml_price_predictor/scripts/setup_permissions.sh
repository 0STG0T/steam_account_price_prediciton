#!/bin/bash

# Make all Python scripts executable
find . -type f -name "*.py" -exec chmod +x {} \;

# Make all shell scripts executable
find . -type f -name "*.sh" -exec chmod +x {} \;

echo "✅ All Python and shell scripts are now executable"
