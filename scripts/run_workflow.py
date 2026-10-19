"""Phase 1 plus Meta. Offline preflight by default; --execute opts into model calls."""

from run_phase1 import main

if __name__ == "__main__":
    main(workflow=True)
