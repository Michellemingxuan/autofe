"""Feature discovery: propose candidate features, then screen them cheaply.

Discovery sits in front of the validation stages. A strategy proposes feature
code; this package makes it safe to run, checks the values it produces, and
scores it on a small sample so the proposer gets feedback within seconds. The
surviving columns are materialised onto the modelling table and handed to the
existing stages, where `leave_one_in` evaluates each one properly and the
verdict gates decide.

Two tiers on purpose: the screen is fast and approximate because a generation
loop needs a signal per round; the verdict is slow and rigorous because it is
the decision that counts.
"""

from discovery.guards import check_finite, check_matrix_finite, check_scale
from discovery.sandbox import (
    CandidateError,
    apply_code,
    assigned_columns,
    referenced_columns,
    validate_code,
    validate_single_column,
)

__all__ = [
    "CandidateError",
    "validate_code",
    "validate_single_column",
    "apply_code",
    "assigned_columns",
    "referenced_columns",
    "check_finite",
    "check_scale",
    "check_matrix_finite",
]
