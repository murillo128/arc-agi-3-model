import { MAX_BYTES, INPUT_OVERHEAD, ResourceLimitError, readerCap } from './errors';
import { readHeader } from './decoder';
import type { DecodedAttempt } from './types';

export const URL_TIMEOUT_MS = 30_000;
export const DECODE_TIMEOUT_MS = 60_000;
export interface Source { kind: 'local' | 'url'; name: string; bytes: number; lastModified?: number }
export type Progress = (received: number, total?: number) => void;

export function remoteURL(input: string): URL {
  let url: URL;
  try { url = new URL(input); } catch { throw new Error('Enter a complete HTTP or HTTPS URL.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Use an HTTP(S) URL without embedded credentials.');
  return url;
}

/** Bounded stream collection shared by local Blob and fetch, with early header refusal. */
export async function readInput(stream: ReadableStream<Uint8Array>, signal: AbortSignal, progress: Progress,
  total?: number, cap = MAX_BYTES): Promise<Uint8Array> {
  readerCap(cap);
  const limit = cap + INPUT_OVERHEAD;
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  let checked = false;
  const header = new Uint8Array(16);
  const cancel = () => { void reader.cancel().catch(() => undefined); };
  signal.addEventListener('abort', cancel, { once: true });
  try {
    if (signal.aborted) throw new DOMException('Load cancelled', 'AbortError');
    if (total !== undefined && total > limit) throw new ResourceLimitError('Compressed input exceeds reader memory limit');
    while (true) {
      const { done, value } = await reader.read();
      if (signal.aborted) throw new DOMException('Load cancelled', 'AbortError');
      if (done) break;
      if (value.length > limit - received) throw new ResourceLimitError('Compressed input exceeds reader memory limit');
      if (received < 16) header.set(value.subarray(0, Math.min(16 - received, value.length)), received);
      received += value.length;
      if (!checked && received >= 16) { readHeader(header, cap); checked = true; }
      chunks.push(value);
      progress(received, total);
    }
    if (!checked) readHeader(header.subarray(0, received), cap);
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return bytes;
  } finally {
    signal.removeEventListener('abort', cancel);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export async function loadLocal(file: File, signal: AbortSignal, progress: Progress): Promise<{ bytes: Uint8Array; source: Source }> {
  const bytes = await readInput(file.stream(), signal, progress, file.size);
  return { bytes, source: { kind: 'local', name: file.name, bytes: file.size, lastModified: file.lastModified } };
}

export async function loadURL(input: string, signal: AbortSignal, progress: Progress): Promise<{ bytes: Uint8Array; source: Source }> {
  const url = remoteURL(input);
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) controller.abort();
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, URL_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal, mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer' });
    if (!response.ok) throw new Error(`URL returned HTTP ${response.status}. Choose another URL or a local file.`);
    if (!response.body) throw new Error('URL response has no readable body.');
    const length = response.headers.get('content-length');
    const total = length && /^\d+$/.test(length) ? Number(length) : undefined;
    const bytes = await readInput(response.body, controller.signal, progress, total);
    return { bytes, source: { kind: 'url', name: url.href, bytes: bytes.length } };
  } catch (error) {
    if (timedOut) throw new Error('URL load exceeded 30 seconds. Retry, choose another URL, or open a local file.');
    if (signal.aborted) throw new DOMException('Load cancelled', 'AbortError');
    if (error instanceof TypeError) throw new Error('Could not fetch the URL: network, offline, or CORS failure. The remote host must allow this viewer origin. Try downloading it and opening the local file.');
    throw error;
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', cancel);
    controller.abort();
  }
}

export function decodeInWorker(bytes: Uint8Array, signal: AbortSignal): Promise<DecodedAttempt> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./decode.worker.ts', import.meta.url), { type: 'module' });
    const finish = (error?: Error, value?: DecodedAttempt) => {
      clearTimeout(timeout);
      signal.removeEventListener('abort', abort);
      worker.terminate(); // Release the WASM heap for every attempt, including failures.
      if (error) reject(error); else resolve(value!);
    };
    const abort = () => finish(new DOMException('Load cancelled', 'AbortError'));
    const timeout = setTimeout(() => finish(new ResourceLimitError('Decode exceeded 60 seconds. Try a smaller recording.')), DECODE_TIMEOUT_MS);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { abort(); return; }
    worker.onerror = () => finish(new Error('Browser decoder failed or ran out of memory. Try a smaller file.'));
    worker.onmessage = (event: MessageEvent<{ value?: DecodedAttempt; error?: string }>) => {
      finish(event.data.error ? new Error(event.data.error) : undefined, event.data.value);
    };
    worker.postMessage(bytes, [bytes.buffer as ArrayBuffer]);
  });
}
