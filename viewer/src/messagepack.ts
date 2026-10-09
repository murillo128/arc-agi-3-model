import { Arc3Error, ResourceLimitError, fail } from './errors';
import type { Value } from './types';

/** Retain FLOAT tags until schema validation; FLOAT 0.0 is not an integer field. */
export class WireFloat { constructor(readonly value: number) {} }
export type WireValue = null | boolean | number | string | Uint8Array | WireFloat | WireValue[] | WireMap;
export interface WireMap { [key: string]: WireValue }
export const MAX_VALUES = 2_000_000;

/** Frozen v1 MessagePack profile, with guards before any container allocation. */
export function unpackPayload(raw: Uint8Array): WireMap {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const utf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  let pos = 0;
  let values = 0;
  function take(size: number): number {
    if (size > raw.length - pos) fail('MessagePack', 'truncated value or excessive declared length');
    const start = pos;
    pos += size;
    return start;
  }
  function uint(width: number): number {
    const offset = take(width);
    if (width === 1) return view.getUint8(offset);
    if (width === 2) return view.getUint16(offset);
    return view.getUint32(offset);
  }
  function str(size: number): string {
    const start = take(size);
    try { return utf8.decode(raw.subarray(start, pos)); }
    catch { return fail('MessagePack', 'invalid UTF-8 STR'); }
  }
  function container(kind: 'map' | 'array', count: number, depth: number): WireValue {
    if (depth >= 64) fail('MessagePack', 'nesting exceeds 64 containers');
    const children = count * (kind === 'map' ? 2 : 1);
    if (children > raw.length - pos) fail('MessagePack', 'container count exceeds remaining input');
    if (children > MAX_VALUES - values) throw new ResourceLimitError('MessagePack exceeds 2,000,000 value budget');
    if (kind === 'array') {
      const result: WireValue[] = [];
      for (let i = 0; i < count; i++) result.push(read(depth + 1));
      return result;
    }
    const result: WireMap = Object.create(null) as WireMap;
    for (let i = 0; i < count; i++) {
      const key = read(depth + 1);
      if (typeof key !== 'string') fail('MessagePack', 'map keys must be STR');
      if (Object.hasOwn(result, key)) fail('MessagePack', 'duplicate map key');
      result[key] = read(depth + 1);
    }
    return result;
  }
  function read(depth: number): WireValue {
    if (++values > MAX_VALUES) throw new ResourceLimitError('MessagePack exceeds 2,000,000 value budget');
    const tag = uint(1);
    if (tag <= 0x7f) return tag;
    if (tag >= 0xe0) return tag - 256;
    if (tag >= 0x80 && tag <= 0x8f) return container('map', tag & 15, depth);
    if (tag >= 0x90 && tag <= 0x9f) return container('array', tag & 15, depth);
    if (tag >= 0xa0 && tag <= 0xbf) return str(tag & 31);
    if (tag === 0xc0) return null;
    if (tag === 0xc2 || tag === 0xc3) return tag === 0xc3;
    if (tag === 0xc4 || tag === 0xc5 || tag === 0xc6) {
      const size = uint(2 ** (tag - 0xc4));
      const start = take(size);
      return raw.subarray(start, pos);
    }
    if (tag === 0xd9 || tag === 0xda || tag === 0xdb) return str(uint(2 ** (tag - 0xd9)));
    if (tag === 0xdc || tag === 0xdd || tag === 0xde || tag === 0xdf) {
      return container(tag < 0xde ? 'array' : 'map', uint(tag % 2 === 0 ? 2 : 4), depth);
    }
    if (tag === 0xca || tag === 0xcb) {
      const offset = take(tag === 0xca ? 4 : 8);
      const value = tag === 0xca ? view.getFloat32(offset) : view.getFloat64(offset);
      if (!Number.isFinite(value)) fail('MessagePack', 'nonfinite number');
      return new WireFloat(value);
    }
    if (tag >= 0xcc && tag <= 0xd3) {
      const signed = tag >= 0xd0;
      const width = 2 ** (tag - (signed ? 0xd0 : 0xcc));
      const offset = take(width);
      let value: number;
      if (width === 8) {
        const big = signed ? view.getBigInt64(offset) : view.getBigUint64(offset);
        if (big < -9007199254740991n || big > 9007199254740991n) fail('MessagePack', 'unsafe integer');
        value = Number(big);
      } else if (signed) {
        value = width === 1 ? view.getInt8(offset) : width === 2 ? view.getInt16(offset) : view.getInt32(offset);
      } else {
        value = width === 1 ? view.getUint8(offset) : width === 2 ? view.getUint16(offset) : view.getUint32(offset);
      }
      return value;
    }
    return fail('MessagePack', 'prohibited extension or invalid tag');
  }
  const result = read(0);
  if (pos !== raw.length) throw new Arc3Error('MessagePack: trailing value or bytes');
  if (!isMap(result)) fail('MessagePack', 'root must be a map');
  return result;
}

export function isMap(value: WireValue | undefined): value is WireMap {
  return value !== null && typeof value === 'object' && !(value instanceof Uint8Array)
    && !(value instanceof WireFloat) && !Array.isArray(value);
}

/** Normalize FLOAT wrappers in place after validation; BIN keeps its zero-copy view. */
export function plainValue(value: WireValue): Value {
  if (value instanceof WireFloat) return value.value;
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) value[i] = plainValue(value[i]!);
  } else if (isMap(value)) {
    for (const key of Object.keys(value)) value[key] = plainValue(value[key]!);
  }
  return value as Value;
}
