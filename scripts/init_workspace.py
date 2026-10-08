"""Create empty runtime directories; never download data or write credentials."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ("runs", "uploads", "cache/tmp", "logs", "configs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    print("Runtime directories initialized.")


if __name__ == "__main__":
    main()
