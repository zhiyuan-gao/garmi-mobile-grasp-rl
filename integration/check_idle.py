"""Require the authorized training run to finish before integration simulation."""
from pathlib import Path
from sim import ensure_training_idle

if __name__ == '__main__':
    ensure_training_idle(Path(__file__).resolve().parents[1])
