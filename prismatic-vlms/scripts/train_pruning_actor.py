# Legacy stub -- refer to staged training scripts
"""
Legacy wrapper retained for backwards compatibility.
Please invoke the staged training scripts directly:
  * Stage 1 (supervised):   python scripts/stage1/train_stage1.py ...
  * Stage 2 (reinforcement): python scripts/stage2/train_stage2.py ...
"""

if __name__ == "__main__":
    raise RuntimeError(
        "This entry point has been deprecated. "
        "Use `scripts/stage1/train_stage1.py` and `scripts/stage2/train_stage2.py` instead."
    )

