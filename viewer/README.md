# Browser .arc3 attempt loader

This static Vite/TypeScript entry point loads the frozen [v1 format](../docs/arc3-format-v1.md)
through a local file dialog, drag/drop, a submitted HTTP(S) URL, or an explicit
`?file=<encoded-https-url>` deep-link. It requires no application backend, SDK,
game execution, Python process, account, database, or proxy at runtime.

The initial screen shows loading progress and recoverable errors. A validated
attempt shows source/provenance, summary, action IDs/parameters and every recorded
observation/frame. Pixels are presented as exact palette indices, including
unknown values. The compact pixel preview displays at most 32 × 32 indices;
the decoder retains complete BIN data and all frames. Observation 0 is the initial
observation; observation 1 is step 0's result. Frame indices are zero-based.
Long text previews are marked as truncated; decoded text remains intact.
This entry point is a loader, rather than a full diagnostic replay interface.

## Run and build

Use Node.js 22.12 or newer and npm:

```sh
cd viewer
npm ci
npm run dev
```

Build and serve the assets with relative paths:

```sh
npm run build
python3 -m http.server 8080 --directory dist
```

Open `http://localhost:8080/`. Any ordinary static host can serve `dist/` at a
nested path, including GitHub Pages. Serve over HTTP(S); opening `index.html`
through `file://` does not provide the module/worker/WASM hosting boundary.
There is no deployment workflow. Build outputs are ignored.

No file is uploaded. Local reads never initiate a remote request. The worker
and WASM are served from the viewer's own bundled static assets, with no CDN or
external decoder service. Offline use requires those application assets to be
available locally; remote URL loads require a network connection. The remote
host must authorize the viewer's origin through CORS. Missing authorization,
network/offline failures, HTTP errors, timeouts and size refusals are reported;
download the recording separately and open it locally when remote access fails.
URL fetches omit cookies/credentials and referrers. URLs embedded in recording
metadata are never resolved. Text is assigned through `textContent`, never HTML.

## Validation and resource limits

`src/decoder.ts` exposes `decodeAttempt(bytes, cap?)`, returning version metadata
and the complete public `Attempt` view model in `src/types.ts`. Optional model,
prediction, learning and diagnostic fields retain their recorded absence; no
reward, model output or default loss is invented. Unknown optional values are
preserved after profile validation; future v1 minors are accepted.

The decoder checks the uint64 header as `BigInt`, the ordinary single-frame
Zstandard envelope, checksum flag, dictionary prohibition, window, exact content
size, block boundary and absence of suffixes. Decompression uses a fixed output
capacity of exactly the declared length; the native decoder verifies the checksum.
Only then does the bounded MessagePack parser validate unique STR map keys,
UTF-8, integer versus FLOAT tags, finite values, lengths, depth and the entire
required/recognized optional schema. Tensor bytes are validated before use,
including unaligned little-endian f16/f32 values. Chronology, pre-action
availability, RESET and summary/termination equations match the Python codec.

The default uncompressed and window cap is **512 MiB**. The API permits a lower
positive cap, never a higher one. Input streams and the decoder also refuse
compressed files larger than the cap plus 1 MiB plus the 16-byte envelope, matching
the Python file-reader policy. MessagePack has a **2,000,000 decoded-value budget**
(including keys and containers), as well as the format's 64-container depth limit.
The browser bounds URL reads to **30 seconds** and worker decode to **60 seconds**.
These additional limits are resource refusals: some otherwise valid files can
exceed them. Total browser memory can exceed the raw-output cap; the compressed
buffer, WASM heap and parsed objects also take space. Smaller files may be needed
on memory-constrained devices. Each load uses a fresh worker, terminated on
success, failure, cancellation, supersession or deadline, releasing its WASM heap.

The sole runtime dependency is pinned `@bokuweb/zstd-wasm@0.0.27`, a WASM build of
upstream libzstd. Its [wrapper source](https://github.com/bokuweb/zstd-wasm/blob/87277ab93c2861d6c265a2ff43c7e7fb14d9a8c5/lib/simple/decompress.ts)
and [build recipe](https://github.com/bokuweb/zstd-wasm/blob/87277ab93c2861d6c265a2ff43c7e7fb14d9a8c5/build.sh)
were inspected for allocation, output-capacity, error handling and checksum use;
tests verify checksum corruption and output overflow rejection through the actual
dependency in Node and Chromium. This is a scoped source audit, not a claim of
independent third-party certification. The app guards properties that libzstd
alone permits, such as concatenated frames. npm's lockfile pins artifact integrity.
The [license notices](public/third-party-notices.txt) are copied into `dist/`.

## Focused local tests

Fixture generation alone needs **Python 3.12** with the repository's `msgpack` and
`pyzstd` dependencies. It imports the conventional Python codec and uses only
synthetic data, without SDK/game execution. For example, from the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'msgpack>=1.1,<2' 'pyzstd>=0.16,<0.20'
cd viewer
npm ci
PYTHON=../.venv/bin/python npm run fixtures
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

On a Linux host missing Chromium shared libraries, use Playwright's documented
`npx playwright install --with-deps chromium` setup. An already installed compatible
Chromium can be selected with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/absolute/path`.

Unit tests own header/framing/decompression/MessagePack/schema/tensor conformance
and bounded stream/URL policy. Fixture expectations come from frozen v1 section 8:
the independent baseline golden plus production-Python one/multi-frame samples,
literal pixel/float/action assertions and malformed vectors constructed directly
with msgpack/pyzstd. The generator verifies every expected accept/reject result
with Python before emitting ignored `.fixtures/` files. It does not regenerate
the repository's independent golden. An undersized output buffer is rejected by
libzstd; decoder error wording is not part of the wire contract.

Playwright owns the real built browser/worker/WASM boundary: local file dialog,
drag/drop, sequence preservation, literal hostile text, user-gated URL success,
HTTPS deep-link, actual CORS refusal, mocked network/HTTP/size failures,
cancellation, supersession, empty frames and recovery. Its only server is Python's
ordinary static asset/fixture server; it serves `dist/` at a nested path. URL
success is a CORS-authorized mock fixture, not evidence that arbitrary remote
hosts allow requests. No CI or push/PR test trigger is added. Run these focused
tests when relevant format, codec, fixture or viewer files change, and for integrated
acceptance; unrelated changes do not require viewer validation.
