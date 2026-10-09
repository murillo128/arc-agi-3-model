"""Errors shared by wire and semantic validation."""

MAX_UNCOMPRESSED_BYTES = 536_870_912


class Arc3Error(ValueError):
    """Malformed, unsupported or semantically inconsistent .arc3 data."""


class Arc3ResourceLimitError(Arc3Error):
    """The requested reader resource cap refuses this input."""
