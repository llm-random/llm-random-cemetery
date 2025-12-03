#!/usr/bin/env python3
"""Check that a config only uses one project folder (excluding src/core)."""

import re
import sys
from pathlib import Path


def check_config_isolation(config_path: str) -> bool:
    """Returns True if config uses at most one project folder."""

    # Read entire config file as text
    text = Path(config_path).read_text()

    # Find all src.<project>.* references (excluding src.core)
    pattern = r'src\.([^.]+)\.'
    matches = re.findall(pattern, text)

    # Filter out 'core', 'definitions' (shared), and internal (_*) modules
    projects = {m for m in matches if m not in ('core', 'definitions') and not m.startswith('_')}

    if len(projects) > 1:
        print(f"⚠️  {config_path} uses multiple projects: {projects}")
        return False

    print(f"✓ {config_path} OK")
    return True


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tiny_remote.yaml"
    sys.exit(0 if check_config_isolation(config_path) else 1)
