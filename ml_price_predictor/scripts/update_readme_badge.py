#!/usr/bin/env python3
"""Update README with latest monitoring badge."""

import argparse
from pathlib import Path
import re

def update_readme_badge(readme_path: str, badge_path: str):
    """Update README.md with the latest monitoring badge."""
    try:
        with open(badge_path) as f:
            badge_content = f.read().strip()

        readme = Path(readme_path)
        content = readme.read_text()

        # Replace badge placeholder or existing badge
        badge_pattern = r'(?<=\n)<monitoring_badge_placeholder>|!\[Monitoring Status\]\(.*?\)'
        if re.search(badge_pattern, content):
            new_content = re.sub(badge_pattern, badge_content, content)
        else:
            # If no placeholder or existing badge found, add after first heading
            new_content = re.sub(r'(#[^\n]+\n)', f'\\1\n{badge_content}\n', content)

        readme.write_text(new_content)
        print("Successfully updated README with new monitoring badge")
    except Exception as e:
        print(f"Error updating README: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Update README with monitoring badge")
    parser.add_argument("--readme", default="README.md",
                      help="Path to README.md file")
    parser.add_argument("--badge", default="monitoring_badge.md",
                      help="Path to badge markdown file")
    args = parser.parse_args()

    update_readme_badge(args.readme, args.badge)

if __name__ == "__main__":
    main()
