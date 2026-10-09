import './style.css';
import { decodeInWorker, loadLocal, loadURL } from './loader';
import { ReplayView } from './replay-view';

function element<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
const fileInput = element<HTMLInputElement>('file-input');
const urlInput = element<HTMLInputElement>('url-input');
const dropZone = element<HTMLDivElement>('drop-zone');
const status = element<HTMLParagraphElement>('status');
const error = element<HTMLParagraphElement>('error');
const progress = element<HTMLProgressElement>('progress');
const cancel = element<HTMLButtonElement>('cancel');
const result = element<HTMLElement>('result');
const replayView = new ReplayView();
let current: AbortController | undefined;

async function open(input: File | string): Promise<void> {
  current?.abort();
  const controller = new AbortController();
  current = controller;
  replayView.clear();
  result.hidden = true;
  error.hidden = true;
  cancel.hidden = false;
  progress.hidden = false;
  progress.removeAttribute('value');
  status.textContent = `Loading ${typeof input === 'string' ? input : input.name}…`;
  const onProgress = (received: number, total?: number) => {
    if (current !== controller) return;
    status.textContent = `Reading ${received.toLocaleString()}${total === undefined ? '' : ` / ${total.toLocaleString()}`} bytes…`;
    if (total !== undefined && total > 0) { progress.max = total; progress.value = received; }
  };
  try {
    const loaded = typeof input === 'string' ? await loadURL(input, controller.signal, onProgress) : await loadLocal(input, controller.signal, onProgress);
    if (current !== controller) return;
    status.textContent = 'Validating header, checksum, schema and tensors…';
    progress.removeAttribute('value');
    const decoded = await decodeInWorker(loaded.bytes, controller.signal);
    if (current !== controller) return;
    replayView.display(decoded, loaded.source);
    status.textContent = 'Attempt validated. Open another recording at any time.';
  } catch (cause) {
    if (current !== controller) return;
    if (controller.signal.aborted) status.textContent = 'Load cancelled. Choose another file or URL.';
    else {
      status.textContent = 'Could not open this recording. Choose another file or URL.';
      error.textContent = cause instanceof Error ? cause.message : String(cause);
      error.hidden = false;
    }
  } finally {
    if (current === controller) { progress.hidden = true; cancel.hidden = true; current = undefined; }
  }
}

fileInput.addEventListener('change', () => { const file = fileInput.files?.[0]; fileInput.value = ''; if (file) void open(file); });
dropZone.addEventListener('keydown', event => { if (event.target === dropZone && ['Enter', ' '].includes(event.key)) { event.preventDefault(); fileInput.click(); } });
dropZone.addEventListener('dragover', event => { event.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
// Prevent the browser from navigating to dropped files anywhere on the page.
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
dropZone.addEventListener('drop', event => {
  event.preventDefault(); dropZone.classList.remove('dragging');
  const files = event.dataTransfer?.files;
  if (files?.length === 1) void open(files[0]!);
  else { error.textContent = 'Choose one .arc3 attempt at a time.'; error.hidden = false; }
});
element('url-form').addEventListener('submit', event => { event.preventDefault(); void open(urlInput.value.trim()); });
cancel.addEventListener('click', () => current?.abort());
const deepLink = new URLSearchParams(location.search).get('file');
if (deepLink !== null) {
  urlInput.value = deepLink;
  if (!deepLink.startsWith('https://')) { error.textContent = 'The ?file= deep-link must contain an HTTPS URL.'; error.hidden = false; }
  else void open(deepLink);
}
