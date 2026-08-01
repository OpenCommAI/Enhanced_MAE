"""Allow ``python -m enhanced_mae`` to invoke the project CLI."""

from enhanced_mae.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
