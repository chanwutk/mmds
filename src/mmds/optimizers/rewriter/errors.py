from ...model import MMDSValidationError


class MMDSRewriteError(MMDSValidationError):
    """Raised when a rewrite cannot be matched, applied, or validated."""
