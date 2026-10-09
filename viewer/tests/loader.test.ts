import { describe, expect, it, vi, afterEach } from 'vitest';
import { readInput, remoteURL, loadURL, URL_TIMEOUT_MS } from '../src/loader';
import { INPUT_OVERHEAD, ResourceLimitError } from '../src/errors';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
function header(size = 1n): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(Buffer.from('41524333010000000100000000000000', 'hex'));
  new DataView(bytes.buffer).setBigUint64(8, size, true);
  return bytes;
}
function stream(chunks: Uint8Array[], onCancel = () => {}): ReadableStream<Uint8Array> {
  return new ReadableStream({ start(controller) { chunks.forEach(chunk => controller.enqueue(chunk)); controller.close(); }, cancel: onCancel });
}

describe('bounded browser input', () => {
  it('reads a split header, reports progress, and retains exact bytes', async () => {
    const bytes = Uint8Array.from([...header(), 1, 2, 3]);
    const progress = vi.fn();
    const read = await readInput(stream([bytes.slice(0, 3), bytes.slice(3, 17), bytes.slice(17)]), new AbortController().signal, progress, bytes.length);
    expect(read).toEqual(bytes);
    expect(progress.mock.calls).toEqual([[3, 19], [17, 19], [19, 19]]);
  });
  it('rejects a bad header before requesting the rest of a remote body', async () => {
    let pulls = 0;
    const cancel = vi.fn();
    const source = new ReadableStream<Uint8Array>({
      pull(controller) { pulls++; controller.enqueue(header(18446744073709551615n)); }, cancel,
    }, { highWaterMark: 0 });
    await expect(readInput(source, new AbortController().signal, () => {})).rejects.toThrow('length');
    expect(pulls).toBe(1);
    expect(cancel).toHaveBeenCalledOnce();
  });
  it('enforces advertised and actual compressed limits, including absent/false lengths', async () => {
    const limit = 32 + INPUT_OVERHEAD;
    await expect(readInput(stream([header()]), new AbortController().signal, () => {}, limit + 1, 32)).rejects.toThrow(ResourceLimitError);
    for (const advertised of [undefined, 16]) {
      await expect(readInput(stream([header(), new Uint8Array(limit)]), new AbortController().signal, () => {}, advertised, 32)).rejects.toThrow(ResourceLimitError);
    }
  });
  it('cancels a stalled reader', async () => {
    const controller = new AbortController();
    const source = new ReadableStream<Uint8Array>({});
    const pending = readInput(source, controller.signal, () => {});
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('user-requested remote loads', () => {
  it('allows HTTP(S) URLs and rejects code, file paths and embedded credentials', () => {
    expect(remoteURL('https://example.org/a.arc3?x=1').href).toBe('https://example.org/a.arc3?x=1');
    expect(remoteURL('http://localhost/a.arc3').protocol).toBe('http:');
    for (const input of ['javascript:alert(1)', 'file:///secret.arc3', 'data:text/html,x', '/relative.arc3', 'https://user:secret@example.org/a']) expect(() => remoteURL(input)).toThrow();
  });
  it('fetches without cookies/referrer or local bytes and does not trust content-length', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(stream([header(), Uint8Array.of(1)]), { headers: { 'content-length': '16' } }));
    vi.stubGlobal('fetch', fetch);
    const loaded = await loadURL('https://example.org/a.arc3', new AbortController().signal, () => {});
    expect(loaded.bytes.length).toBe(17);
    expect(loaded.source).toEqual({ kind: 'url', name: 'https://example.org/a.arc3', bytes: 17 });
    expect(fetch.mock.calls[0]![1]).toMatchObject({ mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer' });
    expect(fetch.mock.calls[0]![1]).not.toHaveProperty('body');
  });
  it('reports network/CORS/offline and HTTP failures without poisoning a later load', async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(new Response(null, { status: 404 }))
      .mockResolvedValueOnce(new Response(header()));
    vi.stubGlobal('fetch', fetch);
    await expect(loadURL('https://example.org/a', new AbortController().signal, () => {})).rejects.toThrow('CORS');
    await expect(loadURL('https://example.org/a', new AbortController().signal, () => {})).rejects.toThrow('HTTP 404');
    expect((await loadURL('https://example.org/b', new AbortController().signal, () => {})).bytes).toEqual(header());
  });
  it('bounds latency through completion of a stalled response body', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new ReadableStream<Uint8Array>({}))));
    const pending = loadURL('https://example.org/slow', new AbortController().signal, () => {});
    const assertion = expect(pending).rejects.toThrow('exceeded 30 seconds');
    await vi.advanceTimersByTimeAsync(URL_TIMEOUT_MS);
    await assertion;
  });
});
