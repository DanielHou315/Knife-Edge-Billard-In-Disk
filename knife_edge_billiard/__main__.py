"""Entry point for ``python -m knife_edge_billiard``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
