import { init, decompress } from '@bokuweb/zstd-wasm';
import { Arc3Error, ResourceLimitError, MAX_BYTES, INPUT_OVERHEAD, fail, readerCap } from './errors';
import { unpackPayload, plainValue } from './messagepack';
import { validateAttempt } from './validation';
import type { Attempt, DecodedAttempt } from './types';

let ready: Promise<void> | undefined;
export function readHeader(bytes: Uint8Array, cap = MAX_BYTES): { minor: number; size: number } {
  readerCap(cap);
  if (bytes.length < 16) fail('ARC3', 'truncated 16-byte header');
  if (bytes[0] !== 65 || bytes[1] !== 82 || bytes[2] !== 67 || bytes[3] !== 51) fail('ARC3', 'invalid magic');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = view.getUint16(4, true);
  if (major !== 1) fail('ARC3', `unsupported major version ${major}`);
  const size = view.getBigUint64(8, true);
  if (size === 0n || size > BigInt(MAX_BYTES)) fail('ARC3', 'invalid uncompressed length (1..512 MiB)');
  if (size > BigInt(cap)) throw new ResourceLimitError('Declared uncompressed length exceeds reader cap');
  return { minor: view.getUint16(6, true), size: Number(size) };
}

/** Inspect the sole frame and its exact boundary before native decompression. */
export function checkFrame(frame: Uint8Array, size: number, cap: number): void {
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  let pos = 0;
  function take(length: number): number {
    if (length > frame.length - pos) fail('Zstandard', 'incomplete frame');
    const start = pos;
    pos += length;
    return start;
  }
  function little(width: number): bigint {
    const start = take(width);
    let value = 0n;
    for (let i = width - 1; i >= 0; i--) value = (value << 8n) | BigInt(frame[start + i]!);
    return value;
  }
  if (view.getUint32(take(4), true) !== 0xfd2fb528) fail('Zstandard', 'expected ordinary frame magic');
  const descriptor = Number(little(1));
  if (!(descriptor & 4)) fail('Zstandard', 'content checksum is required');
  if (descriptor & 8) fail('Zstandard', 'reserved frame header bit');
  const singleSegment = Boolean(descriptor & 32);
  let window = 0;
  if (!singleSegment) {
    const wd = Number(little(1));
    const base = 2 ** (10 + (wd >> 3));
    window = base + (base / 8) * (wd & 7);
  }
  const dictionaryWidth = [0, 1, 2, 4][descriptor & 3]!;
  if (little(dictionaryWidth) !== 0n) fail('Zstandard', 'dictionaries are prohibited');
  const flag = descriptor >> 6;
  const contentWidth = flag === 0 ? (singleSegment ? 1 : 0) : 2 ** flag;
  if (contentWidth) {
    const contentSize = little(contentWidth) + (flag === 1 ? 256n : 0n);
    if (contentSize !== BigInt(size)) fail('Zstandard', 'content size differs from envelope length');
    if (singleSegment) window = Number(contentSize);
  }
  if (window > cap) throw new ResourceLimitError('Zstandard window exceeds reader cap');
  let last = false;
  while (!last) {
    const header = Number(little(3));
    last = Boolean(header & 1);
    const blockType = (header >> 1) & 3;
    const blockSize = header >> 3;
    if (blockType === 3) fail('Zstandard', 'reserved block type');
    if (blockSize > 131072) fail('Zstandard', 'invalid block size');
    take(blockType === 1 ? 1 : blockSize);
  }
  take(4); // The native ZSTD_decompress call verifies these checksum bytes.
  if (pos !== frame.length) fail('Zstandard', 'trailing bytes or concatenated frames');
}

export async function decodeAttempt(bytes: Uint8Array, cap = MAX_BYTES): Promise<DecodedAttempt> {
  const { minor, size } = readHeader(bytes, cap);
  if (bytes.length > cap + INPUT_OVERHEAD) throw new ResourceLimitError('Compressed input exceeds reader memory limit');
  const frame = bytes.subarray(16);
  checkFrame(frame, size, cap);
  ready ??= init();
  await ready;
  let raw: Uint8Array;
  try {
    // Source-audited wrapper around upstream libzstd. Header checks make a
    // declared content size safe; absent content size uses exactly N capacity.
    raw = decompress(frame, { defaultHeapSize: size });
  } catch (error) {
    throw new Arc3Error(`Zstandard decoding/checksum failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (raw.length !== size) fail('ARC3', 'decompressed length differs from envelope length');
  const value = unpackPayload(raw);
  validateAttempt(value, cap);
  return { major: 1, minor, uncompressedBytes: size, attempt: plainValue(value) as unknown as Attempt };
}
