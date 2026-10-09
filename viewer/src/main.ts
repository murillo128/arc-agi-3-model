import './style.css';
import { decodeInWorker, loadLocal, loadURL, type Source } from './loader';
import type { Attempt, DecodedAttempt, Observation } from './types';

function element<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
const fileInput = element<HTMLInputElement>('file-input');
const urlInput = element<HTMLInputElement>('url-input');
const dropZone = element<HTMLDivElement>('drop-zone');
const status = element<HTMLParagraphElement>('status');
const error = element<HTMLParagraphElement>('error');
const progress = element<HTMLProgressElement>('progress');
const cancel = element<HTMLButtonElement>('cancel');
const result = element<HTMLElement>('result');
const observations = element<HTMLInputElement>('observation-select');
const frames = element<HTMLInputElement>('frame-select');
let current: AbortController | undefined;
let attempt: Attempt | undefined;
let activeObservation: Observation | undefined;
function previewText(value: string): string { return value.length > 2000 ? `${value.slice(0, 2000)}… [text preview truncated]` : value; }

function showPixels(): void {
  const tensor = activeObservation!.frames;
  const [, height = 0, width = 0] = tensor.shape;
  const pixels = element<HTMLPreElement>('pixels');
  if (!tensor.shape[0]) { pixels.textContent = 'No frames were returned for this observation.'; return; }
  const index = Math.min(Math.max(Math.trunc(Number(frames.value)) || 0, 0), tensor.shape[0]! - 1);
  frames.value = String(index);
  // This loader's compact preview limits DOM work. The decoded BIN stays exact.
  const rows = Math.min(height, 32), cols = Math.min(width, 32);
  const lines: string[] = [];
  for (let y = 0; y < rows; y++) {
    const line: string[] = [];
    for (let x = 0; x < cols; x++) line.push(String(tensor.data[index * height * width + y * width + x]).padStart(3));
    lines.push(line.join(' '));
  }
  pixels.textContent = lines.join('\n') + (height > rows || width > cols ? '\nPreview limited to the first 32 × 32 indices; complete bytes are retained.' : '');
}

function showObservation(): void {
  const i = Math.min(Math.max(Math.trunc(Number(observations.value)) || 0, 0), attempt!.steps.length);
  observations.value = String(i);
  const step = i === 0 ? undefined : attempt!.steps[i - 1];
  activeObservation = step?.observation ?? attempt!.initial_observation;
  const o = activeObservation;
  const count = o.frames.shape[0]!;
  frames.value = '0';
  frames.max = String(Math.max(0, count - 1));
  frames.disabled = count === 0;
  element('observation-info').textContent = `${previewText(o.state)} · levels ${o.levels_completed}/${o.win_levels} · shape [${o.frames.shape.join(', ')}] · available actions [${o.available_actions.join(', ')}]` +
    (step ? ` · ACTION${step.action.id} ${JSON.stringify(step.action.data)}` : ' · Initial observation');
  showPixels();
}

function display(decoded: DecodedAttempt, source: Source): void {
  attempt = decoded.attempt;
  element('game-id').textContent = previewText(attempt.metadata.game_id);
  element('version').textContent = `ARC3 ${decoded.major}.${decoded.minor}`;
  const metadata = element<HTMLDListElement>('metadata');
  metadata.replaceChildren();
  const entries: [string, string][] = [
    ['Source', `${source.kind === 'local' ? 'Local file' : 'URL'} · ${source.name}`],
    ['Size', `${source.bytes.toLocaleString()} bytes · ${decoded.uncompressedBytes.toLocaleString()} bytes uncompressed`],
    ['Run / session', `${previewText(attempt.metadata.run_id)} / ${previewText(attempt.metadata.session_id)}`],
    ['Attempt / seed', `${attempt.metadata.attempt_index} / ${attempt.metadata.seed}`],
    ['SDK / split', `${previewText(attempt.metadata.sdk_version)} / ${attempt.metadata.source_split}`],
    ['Started', attempt.metadata.started_at],
    ['Model', previewText(attempt.metadata.model_id ?? 'Unavailable (not recorded)')],
    ['Checkpoint', previewText(attempt.metadata.checkpoint_id ?? 'Unavailable (not recorded)')],
    ['Termination', `${attempt.termination.reason} · ${previewText(attempt.termination.final_state)}`],
    ['Summary', `${attempt.summary.real_actions} real actions · ${attempt.summary.levels_completed}/${attempt.summary.win_levels} levels · ${attempt.summary.wall_seconds} seconds`],
  ];
  if (source.lastModified !== undefined) entries.push(['Local file modified', new Date(source.lastModified).toISOString()]);
  for (const [key, value] of entries) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = key; dd.textContent = previewText(value); metadata.append(dt, dd);
  }
  element('sequence-info').textContent = `Initial observation + ${attempt.steps.length} recorded transitions. Frame sequences and palette index bytes are preserved.`;
  observations.value = '0';
  observations.max = String(attempt.steps.length);
  element('recorded-fields').textContent = JSON.stringify({ termination: attempt.termination, summary: attempt.summary }, (_key, value: unknown) => typeof value === 'string' && value.length > 2000 ? `${value.slice(0, 2000)}… [text preview truncated]` : value, 2);
  showObservation();
  result.hidden = false;
}

async function open(input: File | string): Promise<void> {
  current?.abort();
  const controller = new AbortController();
  current = controller;
  attempt = undefined;
  activeObservation = undefined;
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
    display(decoded, loaded.source);
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
observations.addEventListener('change', showObservation);
frames.addEventListener('change', showPixels);
const deepLink = new URLSearchParams(location.search).get('file');
if (deepLink !== null) {
  urlInput.value = deepLink;
  if (!deepLink.startsWith('https://')) { error.textContent = 'The ?file= deep-link must contain an HTTPS URL.'; error.hidden = false; }
  else void open(deepLink);
}
