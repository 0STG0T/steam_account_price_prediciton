#!/usr/bin/env python3
"""Check if all required monitoring dependencies are installed."""

import importlib
import sys

REQUIRED_PACKAGES = {
    'plotly': 'For interactive dashboard generation',
    'kaleido': 'For static image export',
    'pandas': 'For data manipulation',
    'numpy': 'For numerical operations',
    'matplotlib': 'For static plots',
    'seaborn': 'For statistical visualizations'
}

def check_dependencies():
    """Check if all required packages are installed."""
    missing_packages = []

    for package, purpose in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package:<15} - {purpose}")
        except ImportError:
            missing_packages.append(f"✗ {package:<15} - {purpose}")

    if missing_packages:
        print("\nMissing required packages:")
        for pkg in missing_packages:
            print(pkg)
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\nAll monitoring dependencies are installed!")

if __name__ == "__main__":
    check_dependencies()
