"""Baseline attacks, all returning ``AttackResult`` via the shared loss interface.

Each baseline is a faithful, compact reimplementation adapted to the common
runner API so comparisons against ScatterCamo are apples-to-apples.
"""

from .sparse_rs import SparseRSAttack
from .samoo import SAMOOAttack
from .camopatch import CamoPatchAttack

__all__ = ["SparseRSAttack", "SAMOOAttack", "CamoPatchAttack"]
