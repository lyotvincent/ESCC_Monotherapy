"""CD8 positive-cell response-group analysis."""

import sys
from pathlib import Path


ANALYZE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ANALYZE_DIR / "CD8"
sys.path.insert(0, str(ANALYZE_DIR / "CCR7"))

from CCR7 import main as run_marker_analysis


def has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def main() -> int:
    if not has_option(sys.argv[1:], "--marker"):
        sys.argv.extend(["--marker", "CD8"])
    if not has_option(sys.argv[1:], "--out-dir"):
        sys.argv.extend(["--out-dir", str(DEFAULT_OUT_DIR)])
    return run_marker_analysis()


if __name__ == "__main__":
    raise SystemExit(main())
