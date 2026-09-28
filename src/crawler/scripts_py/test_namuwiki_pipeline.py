"""Safe offline regression entrypoint. No paid API calls or raw-meta rewrites."""
from pathlib import Path
import unittest


def main():
    project_root = Path(__file__).resolve().parents[3]
    suite = unittest.defaultTestLoader.discover(str(project_root / "tests/context"))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
