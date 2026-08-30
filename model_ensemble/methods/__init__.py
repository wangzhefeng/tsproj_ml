"""Fusion method implementations (v4 §3). Each module is a pure algorithm:
methods never split folds, train members, or read files."""

from model_ensemble.methods import averaging, linear_blending, stacking, weighted

__all__ = ["averaging", "linear_blending", "stacking", "weighted"]
