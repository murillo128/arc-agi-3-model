import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeAttempt, readHeader } from '../src/decoder';
import { unpackPayload, plainValue, MAX_VALUES } from '../src/messagepack';
import { tensorElement } from '../src/validation';
import { MAX_BYTES, ResourceLimitError } from '../src/errors';

interface Vector { name: string; hex: string; valid: boolean; expected?: unknown; cap?: number; error?: string }
const vectors = JSON.parse(readFileSync(new URL('../.fixtures/vectors.json', import.meta.url), 'utf8')) as Vector[];
function jsonValue(value: unknown): unknown {
  if (value instanceof Uint8Array) return { $bin_hex: Buffer.from(value).toString('hex') };
  if (Array.isArray(value)) return value.map(jsonValue);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jsonValue(v)]));
  return value;
}

describe('frozen v1 vectors checked against Python', () => {
  for (const v of vectors) it(v.name, async () => {
    const bytes = new Uint8Array(Buffer.from(v.hex, 'hex'));
    if (v.valid) {
      const decoded = await decodeAttempt(bytes, v.cap);
      expect(jsonValue(decoded.attempt)).toEqual(v.expected);
      expect(decoded.major).toBe(1);
    } else await expect(decodeAttempt(bytes, v.cap)).rejects.toThrow(new RegExp(v.error ?? '.'));
  });
});

it('preserves one-frame literals and absent model diagnostics', async () => {
  const v = vectors.find(v => v.name === 'python-one-frame')!;
  const { attempt } = await decodeAttempt(new Uint8Array(Buffer.from(v.hex, 'hex')));
  expect([...attempt.initial_observation.frames.data]).toEqual([0, 1, 2, 3]);
  expect([...attempt.steps[0]!.observation.frames.data]).toEqual([3, 2, 1, 0]);
  expect(attempt.metadata).not.toHaveProperty('model_id');
  expect(attempt.steps[0]).not.toHaveProperty('prediction');
  expect(attempt.steps[0]).not.toHaveProperty('reward');
});

it('preserves repeats, unknown palette bytes, click parameters and little-endian diagnostics', async () => {
  const v = vectors.find(v => v.name === 'python-multi-frame-diagnostics')!;
  const { attempt } = await decodeAttempt(new Uint8Array(Buffer.from(v.hex, 'hex')));
  const step = attempt.steps[0]!;
  expect(step.observation.frames.shape).toEqual([3, 2, 2]);
  expect([...step.observation.frames.data]).toEqual([3, 2, 1, 0, 3, 2, 1, 0, 255, 16, 7, 0]);
  expect(step.action).toEqual({ id: 6, data: { x: 63, y: 0, extra: -2 } });
  expect(tensorElement('f32', step.prediction!.frames!.data, 1)).toBe(1);
  expect(tensorElement('f16', step.prediction!.latent!.data, 0)).toBe(1);
  expect(tensorElement('f16', step.prediction!.latent!.data, 1)).toBe(-2);
  expect(step.notes?.[0]).toBe('π');
});

it('reads uint64 without truncation and allows the default cap boundary', () => {
  const header = new Uint8Array(Buffer.from('41524333010000000000002000000000', 'hex'));
  expect(readHeader(header).size).toBe(MAX_BYTES);
  new DataView(header.buffer).setBigUint64(8, 4294967297n, true);
  expect(() => readHeader(header)).toThrow('length');
  expect(() => readHeader(header, MAX_BYTES + 1)).toThrow('Reader cap');
});

it('refuses a bounded decoded-object budget before allocating an enormous list', () => {
  const raw = new Uint8Array(5 + MAX_VALUES);
  raw[0] = 0xdd;
  new DataView(raw.buffer).setUint32(1, MAX_VALUES);
  expect(() => unpackPayload(raw)).toThrow(ResourceLimitError);
});

it('counts container depth including the top-level map and preserves BOM and safe special keys', () => {
  const maxDepth = Uint8Array.from([0x81, 0xa1, 120, ...Array<number>(63).fill(0x91), 0xc0]);
  expect(() => unpackPayload(maxDepth)).not.toThrow();
  const special = Uint8Array.from([0x81, 0xa9, ...Buffer.from('__proto__'), 0xa4, 0xef, 0xbb, 0xbf, 120]);
  const value = plainValue(unpackPayload(special));
  expect(Object.getPrototypeOf(value)).toBe(null);
  expect((value as Record<string, unknown>).__proto__).toBe('\uFEFFx');
});

it('handles unaligned float tensors, subnormals and signed zero', () => {
  const data = Uint8Array.from([255, 0, 60, 1, 0, 0, 128]).subarray(1);
  expect(tensorElement('f16', data, 0)).toBe(1);
  expect(tensorElement('f16', data, 1)).toBe(2 ** -24);
  expect(Object.is(tensorElement('f16', data, 2), -0)).toBe(true);
});

it.each([
  ['ccff', 255], ['cd0100', 256], ['ce01000000', 16777216], ['cf001fffffffffffff', Number.MAX_SAFE_INTEGER],
  ['d080', -128], ['d1ff00', -256], ['d2ff000000', -16777216], ['d3ffe0000000000001', -Number.MAX_SAFE_INTEGER],
  ['ca3f800000', 1], ['cb3ff0000000000000', 1],
])('decodes big-endian numeric tag %s independently', (hex, expected) => {
  const value = plainValue(unpackPayload(Uint8Array.from([0x81, 0xa1, 120, ...Buffer.from(hex, 'hex')])));
  expect((value as Record<string, unknown>).x).toBe(expected);
});

it('enforces the compressed-input cap before native decoding', async () => {
  const vector = vectors.find(v => v.name === 'independent-golden')!;
  const valid = new Uint8Array(Buffer.from(vector.hex, 'hex'));
  const cap = readHeader(valid).size;
  const oversized = new Uint8Array(cap + 1_048_592 + 1);
  oversized.set(valid);
  await expect(decodeAttempt(oversized, cap)).rejects.toThrow('Compressed input');
});
