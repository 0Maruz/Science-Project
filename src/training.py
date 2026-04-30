"""Backwards-compat shim. The real entry point is now src/train.py.

Kept so that existing docs / cron jobs invoking `python training.py` still work.
"""

from train import main

if __name__ == "__main__":
    main()
