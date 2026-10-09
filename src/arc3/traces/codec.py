"""Strict, dependency-bounded .arc3 v1 codec. See docs/arc3-format-v1.md."""

import os
from pathlib import Path
import struct
import tempfile
from typing import cast

import msgpack
import pyzstd

from ._errors import Arc3Error, Arc3ResourceLimitError, MAX_UNCOMPRESSED_BYTES
from ._messagepack import unpack_payload
from ._validation import validate_attempt
from .types import Attempt

_HEADER = struct.Struct("<4sHHQ")
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_FILE_OVERHEAD_ALLOWANCE = 1_048_576


def _cap(value: int) -> int:
    if type(value) is not int or not 0 < value <= MAX_UNCOMPRESSED_BYTES:
        raise Arc3Error("max_uncompressed_bytes must be positive and no greater than 512 MiB")
    return value


def _header(data: bytes, cap: int) -> int:
    if len(data) < _HEADER.size:
        raise Arc3Error("truncated 16-byte .arc3 header")
    magic, major, _minor, size = _HEADER.unpack_from(data)
    if magic != b"ARC3":
        raise Arc3Error("invalid .arc3 magic")
    if major != 1:
        raise Arc3Error(f"unsupported .arc3 major version {major}")
    # All forward minor values are accepted after validating the known v1 fields.
    if size == 0 or size > MAX_UNCOMPRESSED_BYTES:
        raise Arc3Error("invalid uncompressed length: expected 1..536870912 bytes")
    if size > cap:
        raise Arc3ResourceLimitError("declared uncompressed length exceeds reader cap")
    return size


def _frame_header(frame: memoryview, size: int, cap: int) -> None:
    if len(frame) < 6 or frame[:4] != _ZSTD_MAGIC:
        raise Arc3Error("expected one ordinary Zstandard frame")
    descriptor = frame[4]
    if not descriptor & 4:
        raise Arc3Error("Zstandard content checksum is required")
    try:
        info = pyzstd.get_frame_info(frame[:18])
    except pyzstd.ZstdError as exc:
        raise Arc3Error(f"invalid Zstandard frame header: {exc}") from exc
    if info.dictionary_id:
        raise Arc3Error("Zstandard dictionaries are prohibited")
    if info.decompressed_size is not None and info.decompressed_size != size:
        raise Arc3Error("Zstandard content size differs from envelope length")
    if descriptor & 32:  # Single segment: its content size is the window size.
        window = info.decompressed_size
    else:
        window_descriptor = frame[5]
        base = 1 << (10 + (window_descriptor >> 3))
        window = base + (base >> 3) * (window_descriptor & 7)
    if window > cap:
        raise Arc3ResourceLimitError("Zstandard window size exceeds reader cap")


def encode_attempt(attempt: Attempt) -> bytes:
    """Validate plain wire values and emit a complete v1.0 file as bytes.

    Optional/unknown keys remain intact. This checks syntax and consistency;
    the caller owns provenance, diagnostic capture timing and sanitization.
    """
    validate_attempt(attempt, MAX_UNCOMPRESSED_BYTES)
    try:
        raw = msgpack.packb(attempt, use_bin_type=True, strict_types=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise Arc3Error(f"MessagePack encoding failed: {exc}") from exc
    if not 0 < len(raw) <= MAX_UNCOMPRESSED_BYTES:
        raise Arc3ResourceLimitError("encoded payload exceeds 512 MiB")
    try:
        frame = pyzstd.compress(raw, {
            pyzstd.CParameter.checksumFlag: 1,
            pyzstd.CParameter.contentSizeFlag: 1,
            pyzstd.CParameter.windowLog: 29,
        })
    except pyzstd.ZstdError as exc:
        raise Arc3Error(f"Zstandard encoding failed: {exc}") from exc
    return _HEADER.pack(b"ARC3", 1, 0, len(raw)) + frame


def decode_attempt(data: bytes, *, max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Attempt:
    """Accept only a complete single-frame file; return no partial attempt.

    The cap may be lowered, never raised beyond v1's maximum. Later minor
    versions and structurally valid unknown optional keys are retained.
    """
    cap = _cap(max_uncompressed_bytes)
    if type(data) is not bytes:
        raise Arc3Error(".arc3 input must be bytes")
    size = _header(data, cap)
    frame = memoryview(data)[_HEADER.size:]
    _frame_header(frame, size, cap)
    try:
        # Header inspection enforces the exact cap, including non-power-of-two
        # lower caps. The native window limit is a second guard (minimum 1 KiB).
        decoder = pyzstd.ZstdDecompressor(option={
            pyzstd.DParameter.windowLogMax: max(10, (cap - 1).bit_length()),
        })
        raw = decoder.decompress(frame, max_length=size)
        if not decoder.eof:
            raise Arc3Error("incomplete Zstandard frame or output exceeds declared length")
        if decoder.unused_data:
            raise Arc3Error("trailing bytes or concatenated Zstandard frames")
        if len(raw) != size:
            raise Arc3Error("decompressed length differs from envelope length")
    except pyzstd.ZstdError as exc:
        raise Arc3Error(f"Zstandard decoding/checksum failed: {exc}") from exc
    attempt = unpack_payload(raw)
    validate_attempt(attempt, cap)
    return cast(Attempt, attempt)


def write_attempt(path: str | os.PathLike[str], attempt: Attempt) -> None:
    """Encode first, then fsync a same-directory temporary and atomically replace.

    Any failure before replacement leaves the destination absent or unchanged.
    I/O errors propagate as OSError; temporary files are removed on failure.
    """
    encoded = encode_attempt(attempt)
    target = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_attempt(path: str | os.PathLike[str], *,
                 max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Attempt:
    """Read and validate locally without resolving any payload metadata.

    In addition to the raw/window cap, this convenience reader limits compressed
    file bytes to cap + 1 MiB + the 16-byte envelope. Pathological valid frames
    with more framing overhead receive a resource refusal, not a malformed-file
    claim. decode_attempt can validate such bytes when managed by the caller.
    """
    cap = _cap(max_uncompressed_bytes)
    with Path(path).open("rb") as stream:
        header = stream.read(_HEADER.size)
        _header(header, cap)  # Refuse bogus uint64 lengths before reading the body.
        frame_limit = cap + _FILE_OVERHEAD_ALLOWANCE
        frame = stream.read(frame_limit + 1)
        if len(frame) > frame_limit:
            raise Arc3ResourceLimitError("compressed file exceeds read_attempt memory limit")
    return decode_attempt(header + frame, max_uncompressed_bytes=cap)
