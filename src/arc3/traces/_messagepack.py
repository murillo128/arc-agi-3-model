"""Allocation guards for the v1 MessagePack profile, followed by msgpack decoding.

This scan does not deserialize values. It checks lengths against remaining bytes
and container depth before the installed parser can allocate Python containers.
"""

import msgpack

from ._errors import Arc3Error


def unpack_payload(raw: bytes) -> dict:
    end = len(raw)

    def advance(pos: int, size: int) -> int:
        if size > end - pos:
            raise Arc3Error("MessagePack: truncated value or excessive declared length")
        return pos + size

    def length(pos: int, width: int) -> tuple[int, int]:
        stop = advance(pos, width)
        return int.from_bytes(raw[pos:stop], "big"), stop

    def scan(pos: int, depth: int) -> int:
        advance(pos, 1)
        tag = raw[pos]
        pos += 1
        kind = "scalar"
        count = 0
        if tag <= 0x7f or tag >= 0xe0 or tag in (0xc0, 0xc2, 0xc3):
            return pos
        if 0x80 <= tag <= 0x8f:
            kind, count = "map", tag & 15
        elif 0x90 <= tag <= 0x9f:
            kind, count = "array", tag & 15
        elif 0xa0 <= tag <= 0xbf:
            return advance(pos, tag & 31)
        elif tag in (0xc4, 0xc5, 0xc6, 0xd9, 0xda, 0xdb):
            width = {0xc4: 1, 0xc5: 2, 0xc6: 4, 0xd9: 1, 0xda: 2, 0xdb: 4}[tag]
            size, pos = length(pos, width)
            return advance(pos, size)
        elif tag in (0xdc, 0xdd, 0xde, 0xdf):
            kind = "array" if tag in (0xdc, 0xdd) else "map"
            count, pos = length(pos, 2 if tag in (0xdc, 0xde) else 4)
        elif tag in (0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3):
            size = {0xca: 4, 0xcb: 8, 0xcc: 1, 0xcd: 2, 0xce: 4, 0xcf: 8,
                    0xd0: 1, 0xd1: 2, 0xd2: 4, 0xd3: 8}[tag]
            return advance(pos, size)
        else:
            raise Arc3Error("MessagePack: prohibited extension or invalid tag")

        if depth >= 64:
            raise Arc3Error("MessagePack: nesting exceeds 64 containers")
        children = count * (2 if kind == "map" else 1)
        if children > end - pos:
            raise Arc3Error("MessagePack: container count exceeds remaining input")
        for _ in range(children):
            pos = scan(pos, depth + 1)
        return pos

    if scan(0, 0) != end:
        raise Arc3Error("MessagePack: trailing value or bytes")

    def unique_map(pairs: list[tuple]) -> dict:
        result = {}
        for key, value in pairs:
            if type(key) is not str:
                raise Arc3Error("MessagePack: map keys must be STR")
            if key in result:
                raise Arc3Error(f"MessagePack: duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = msgpack.unpackb(
            raw, raw=False, strict_map_key=False, object_pairs_hook=unique_map,
            max_str_len=end, max_bin_len=end, max_array_len=end, max_map_len=end // 2,
            max_ext_len=0,
        )
    except (ValueError, UnicodeError, msgpack.UnpackException) as exc:
        raise Arc3Error(f"MessagePack: {exc}") from exc
    if type(value) is not dict:
        raise Arc3Error("MessagePack: root must be a map")
    return value
