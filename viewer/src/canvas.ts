import type { Observation } from './types';

// ARC Prize SDK display palette; pinned provenance and MIT notice in third-party-notices.txt.
export const PALETTE = ['ffffff', 'cccccc', '999999', '666666', '333333', '000000', 'e53aa3', 'ff7bcc',
  'f93c31', '1e93ff', '88d8f1', 'ffdc00', 'ff851b', '921231', '4fcc30', 'a356d6'];
const colors = PALETTE.map(hex => [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4), 16)]);

export function drawObservation(canvas: HTMLCanvasElement, observation: Observation | undefined, frame: number): string {
  canvas.hidden = true;
  canvas.width = canvas.height = 1; // Release the previous bitmap, including on missing/oversized frames.
  if (!observation) return 'No pre-action observation for the initial view.';
  const [count = 0, height = 0, width = 0] = observation.frames.shape;
  if (!count) return 'No frames were returned for this observation.';
  // A valid recording can exceed practical browser bitmap limits. Refuse explicitly, never crop it.
  if (width > 16384 || height > 16384 || width * height > 16_777_216) return 'Frame exceeds the viewer bitmap limit (16,384 per axis / 16 Mi pixels). Raw indices remain available.';
  try {
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) return 'Canvas 2D is unavailable in this browser.';
    context.imageSmoothingEnabled = false;
    const bitmap = context.createImageData(width, height);
    const offset = Math.min(Math.max(0, frame), count - 1) * width * height;
    const unknown = new Set<number>();
    for (let i = 0; i < width * height; i++) {
      const value = observation.frames.data[offset + i]!;
      const color = colors[value];
      if (!color) unknown.add(value);
      // A distinct viewer-only fallback, explicitly labelled with the exact indices below.
      bitmap.data.set(color ?? [0, 255, 255], i * 4);
      bitmap.data[i * 4 + 3] = 255;
    }
    context.putImageData(bitmap, 0, 0);
    canvas.hidden = false;
    canvas.style.aspectRatio = `${width} / ${height}`;
    return unknown.size ? `Unknown palette indices: ${[...unknown].sort((a, b) => a - b).join(', ')} (cyan fallback; raw bytes preserved).` : '';
  } catch { return 'This browser could not allocate the frame bitmap. Raw indices remain available.'; }
}

export function pixelPreview(observation: Observation, frame: number): string {
  const [count = 0, height = 0, width = 0] = observation.frames.shape;
  if (!count) return 'No frames were returned for this observation.';
  const lines: string[] = [];
  for (let y = 0; y < Math.min(height, 32); y++) {
    const row: string[] = [];
    for (let x = 0; x < Math.min(width, 32); x++) row.push(String(observation.frames.data[frame * height * width + y * width + x]).padStart(3));
    lines.push(row.join(' '));
  }
  return lines.join('\n') + (height > 32 || width > 32 ? '\nText preview limited to 32 × 32; Canvas renders the complete frame within bitmap limits.' : '');
}
