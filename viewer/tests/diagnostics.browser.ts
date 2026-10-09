import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

async function open(page: Page, name = 'diagnostics', navigate = true): Promise<void> {
  if (navigate) await page.goto('./');
  await page.locator('#file-input').setInputFiles({ name: `${name}.arc3`, mimeType: 'application/octet-stream', buffer: readFileSync(new URL(`../.fixtures/${name}.arc3`, import.meta.url)) });
  await expect(page.getByRole('status')).toContainText('Attempt validated');
}
async function choose(page: Page, id: string, value: number): Promise<void> {
  await page.locator(`#${id}`).fill(String(value)); await page.locator(`#${id}`).dispatchEvent('change');
}
async function rgba(page: Page, id: string): Promise<number[]> {
  return page.locator(`#${id}`).evaluate((element: HTMLCanvasElement) => [...element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data]);
}
const black = [0, 0, 0, 255], red = [249, 60, 49, 255];

test('sequence predictions track selected frames with literal 0/4 and 2/4 pixel oracles', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 1);
  await expect(page.locator('#prediction-info')).toContainText('post_action_sequence · real frame 0');
  await expect(page.locator('#difference-info')).toContainText('0 (0/4 pixels; 0.00%)');
  expect(await rgba(page, 'difference-canvas')).toEqual([...black, ...black, ...black, ...black]);
  expect(await rgba(page, 'prediction-canvas')).toEqual(await rgba(page, 'observation-canvas'));
  await choose(page, 'frame-select', 1);
  await expect(page.locator('#difference-info')).toContainText('0.5 (2/4 pixels; 50.00%)');
  expect(await rgba(page, 'difference-canvas')).toEqual([...black, ...red, ...black, ...red]);
  // Exact palette [3,9,1,9], independent of the code that calculates disagreement.
  expect(await rgba(page, 'prediction-canvas')).toEqual([102, 102, 102, 255, 30, 147, 255, 255, 204, 204, 204, 255, 30, 147, 255, 255]);
});

test('last-frame predictions appear only at the same instant and floats compare before display rounding', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 2);
  await expect(page.locator('#prediction-info')).toContainText('aligns only with real frame 1');
  await expect(page.locator('#prediction-canvas')).toBeHidden();
  await expect(page.locator('#difference-canvas')).toBeHidden();
  await choose(page, 'frame-select', 1);
  await expect(page.locator('#prediction-info')).toContainText('Float palette estimates rounded for display only');
  await expect(page.locator('#difference-info')).toContainText('0.25 (1/4 pixels; 25.00%)');
  expect(await rgba(page, 'difference-canvas')).toEqual([...black, ...red, ...black, ...black]);
  expect(await rgba(page, 'prediction-canvas')).toEqual(await rgba(page, 'observation-canvas'));
  await expect(page.locator('#prediction-values')).toContainText('[3,2.25,1,0]');
  await expect(page.locator('#recorded-errors')).toContainText('MAE (palette-index units): 0.0625');
  await expect(page.locator('#latent-info')).toHaveText('f16 [2] · 2 elements\nL2 norm: 5 (model-specific units)');
  await choose(page, 'frame-select', 0); await expect(page.locator('#prediction-canvas')).toBeHidden();
  await expect(page.locator('#prediction-canvas')).toHaveAttribute('width', '1');
});

test('scores, entropy, public levels, notes and inferred events have honest labels', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 1);
  await expect(page.locator('#score-heading')).toHaveText('Logit (recorded)');
  await expect(page.locator('#score-rows tr')).toHaveText(['ACTION1 · {}-2.5', 'ACTION6 (1, 0) · {"x":1,"y":0}4']);
  await expect(page.locator('#chosen-action')).toHaveText('ACTION1 · {}');
  await expect(page.locator('#uncertainty-info')).toContainText('Action entropy: 0.75 (nats)');
  await expect(page.locator('#level-progress-info')).toHaveText('1 / 2 public levels completed · counter 0 → 1');
  await expect(page.locator('#inferred-event')).toContainText('Inferred by viewer from public observations:');
  await expect(page.locator('#rationale-info')).toHaveText('Recorded plain-text notes: Not recorded.');
  await choose(page, 'observation-select', 2);
  await expect(page.locator('#score-heading')).toHaveText('Probability (recorded)');
  await expect(page.locator('#score-rows tr td:last-child')).toHaveText(['0.25', '0.5']);
  await expect(page.locator('#chosen-action')).toContainText('ACTION6 (1, 0)');
  await expect(page.locator('#selected-value')).toContainText('value estimate: -0.25');
  await expect(page.locator('#rationale-info')).toContainText('<img src=x onerror=alert(1)> is plain text');
  await expect(page.locator('#rationale-info img')).toHaveCount(0);
  await choose(page, 'observation-select', 4);
  await expect(page.locator('#score-heading')).toHaveText('Value (recorded)');
  await expect(page.locator('#score-rows tr td:last-child')).toHaveText(['-1', '2']);
  await expect(page.locator('#uncertainty-info')).toContainText('Action entropy: 0 (nats)');
  await expect(page.locator('#level-progress-info')).toContainText('2 / 2 public levels completed');
  await expect(page.locator('#level-progress')).toHaveAttribute('value', '2');
  await expect(page.locator('#inferred-event')).toContainText('Frames unchanged · Level counter 1 → 2 · State WIN · Terminal');
});

test('uncertainty, loss and timing chart selection preserves missing samples and inclusive filters', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 2);
  await expect(page.locator('#chart-info')).toContainText('steps 0–3 · 3 recorded samples; 1 missing');
  await expect(page.locator('#chart-values tr')).toHaveText(['00.2', '10.5', '2Not recorded', '30']);
  await expect(page.locator('#diagnostic-chart .chart-point, #diagnostic-chart .chart-selected')).toHaveCount(3);
  await expect(page.locator('#diagnostic-chart .chart-line')).toHaveCount(1); // no bridge over step 2
  await expect(page.locator('#diagnostic-chart .chart-selected')).toHaveCount(1);
  await page.locator('#metric-select').selectOption('loss');
  await expect(page.locator('#chart-values tr')).toHaveText(['00.25', '10.125', '2Not recorded', '3-0.5']);
  await choose(page, 'chart-from', 1); await choose(page, 'chart-to', 2);
  await expect(page.locator('#chart-values tr')).toHaveText(['10.125', '2Not recorded']);
  await expect(page.locator('#chart-info')).toContainText('steps 1–2 · 1 recorded samples; 1 missing');
  await page.locator('#metric-select').selectOption('timing.learning_ms');
  await expect(page.locator('#chart-values tr')).toHaveText(['10', '2Not recorded']);
  await choose(page, 'chart-from', 2);
  await expect(page.locator('#chart-info')).toContainText('0 recorded samples; 1 missing');
  await expect(page.locator('#diagnostic-chart circle')).toHaveCount(0);
  await choose(page, 'chart-to', 1);
  await expect(page.locator('#chart-info')).toContainText('Choose an inclusive range');
});

test('partial diagnostics, latent-only and mismatched sequence predictions never invent a visual diff', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 3);
  await expect(page.locator('#prediction-info')).toContainText('Not recorded');
  await expect(page.locator('#uncertainty-info')).toContainText('Prediction uncertainty: Not recorded');
  await expect(page.locator('#uncertainty-info')).toContainText('Action entropy: Not recorded');
  await expect(page.locator('#learning-info')).toContainText('Learning loss (recorded objective): Not recorded');
  await expect(page.locator('#inferred-event')).toContainText('Frames changed');
  await open(page, 'latent-only', false); await choose(page, 'observation-select', 1);
  await expect(page.locator('#prediction-info')).toContainText('model-specific latent decoder is unavailable');
  await expect(page.locator('#difference-canvas')).toBeHidden();
  await expect(page.locator('#latent-info')).toContainText('4 elements');
  await open(page, 'prediction-mismatch', false); await choose(page, 'observation-select', 1);
  await expect(page.locator('#prediction-info')).toContainText('predicted shape [1, 2, 2] differs from target [2, 2, 2]');
  await expect(page.locator('#difference-info')).toContainText('no comparable decoded visual prediction');
  await expect(page.locator('#difference-canvas')).toBeHidden();
});

test('replacing a populated trace with a baseline and visiting RESET clears optional data', async ({ page }) => {
  await open(page); await choose(page, 'observation-select', 2); await choose(page, 'frame-select', 1);
  await open(page, 'replay-levels', false); await choose(page, 'observation-select', 1);
  for (const id of ['prediction-canvas', 'difference-canvas', 'score-table']) await expect(page.locator(`#${id}`)).toBeHidden();
  await expect(page.locator('#difference-canvas')).toHaveAttribute('width', '1');
  await expect(page.locator('#metric-select')).toBeDisabled();
  await expect(page.locator('#chart-values tr')).toHaveCount(0);
  await expect(page.locator('#learning-info')).toContainText('Optimizer updates: Not recorded');
  await expect(page.locator('#latent-info')).toHaveText('Not recorded.');
  await open(page, 'closing-reset', false); await choose(page, 'observation-select', 2);
  await expect(page.locator('#chosen-action')).toContainText('RESET · executed closing boundary');
  await expect(page.locator('#score-table')).toBeHidden();
  await expect(page.locator('#prediction-info')).toContainText('Not recorded');
});

test('diagnostic interactions after opening stay local', async ({ page }) => {
  await open(page); const requests: string[] = [];
  page.on('request', request => requests.push(request.url()));
  await choose(page, 'observation-select', 2); await choose(page, 'frame-select', 1);
  await page.locator('#metric-select').selectOption('loss'); await choose(page, 'chart-from', 1);
  expect(requests).toEqual([]);
});

for (const [name, width, height] of [['desktop', 1440, 1100], ['narrow', 390, 844]] as const) {
  test(`populated diagnostic panels ${name} layout`, async ({ page }) => {
    await page.setViewportSize({ width, height }); await open(page); await choose(page, 'observation-select', 2); await choose(page, 'frame-select', 1);
    await expect(page.locator('#difference-info')).toContainText('0.25 (1/4 pixels');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page).toHaveScreenshot(`diagnostics-${name}.png`, { fullPage: true, animations: 'disabled', maxDiffPixels: 100 });
  });
}
