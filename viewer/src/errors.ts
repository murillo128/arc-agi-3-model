export const MAX_BYTES = 536_870_912;
export const INPUT_OVERHEAD = 1_048_592;
export class Arc3Error extends Error {}
export class ResourceLimitError extends Arc3Error {}
export function fail(path: string, message: string): never { throw new Arc3Error(`${path}: ${message}`); }
export function readerCap(cap: number): number {
  if (!Number.isSafeInteger(cap) || cap <= 0 || cap > MAX_BYTES) fail('Reader cap', 'expected 1..512 MiB');
  return cap;
}
