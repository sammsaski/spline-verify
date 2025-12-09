"""Verification module: safety verification pipeline."""

from .objective import compute_objective, ObjectiveFunction
from .verifier import SafetyVerifier, VerificationResult, VerificationStatus
from .error_bounds import ErrorBudget
from .switching import (
    SwitchingVerifier,
    SwitchingRegionClassifier,
    extract_crossing_labels,
    extract_crossing_sequence,
)

__all__ = [
    # Core verification
    "compute_objective",
    "ObjectiveFunction",
    "SafetyVerifier",
    "VerificationResult",
    "VerificationStatus",
    "ErrorBudget",
    # Switching systems
    "SwitchingVerifier",
    "SwitchingRegionClassifier",
    "extract_crossing_labels",
    "extract_crossing_sequence",
]
