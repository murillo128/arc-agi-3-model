`arc3-v1-baseline.hex` is a frozen, tiny synthetic wire fixture for baseline B in
`docs/arc3-format-v1.md` section 8. It was assembled independently of the
production encoder from the literal baseline values using MessagePack 1.2.3
(`use_bin_type=True`), pyzstd 0.19.1 (checksum and content size enabled), and a
little-endian `ARC3`, major 1, minor 0, payload-length envelope. Whitespace in
the hex file is only a text storage convenience. It contains no SDK game data.

Tests compare its decoded map and tensor values with the specification's literal
values. Other vectors build wire bytes directly with the installed libraries,
bypassing production encoding, and separately verify semantic round-trip.
Compression/map ordering is not canonical, so different interoperable encodings
need not reproduce this fixture's compressed bytes. Do not refresh it to match
an implementation change.
