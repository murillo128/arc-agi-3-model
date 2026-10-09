import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

// These three checked-in goldens are emitted by the Python runtime/SDK adapter/
// EvaluationRecorder, rather than assembled directly in the browser test.
function fixture(name: string): Buffer {
  return readFileSync(new URL(`../../tests/fixtures/${name}.arc3`, import.meta.url));
}
async function drop(page: Page, name: string): Promise<void> {
  const transfer = await page.evaluateHandle((bytes: number[]) => {
    const data = new DataTransfer();
    data.items.add(new File([new Uint8Array(bytes)], 'recorder.arc3', { type: 'application/octet-stream' }));
    return data;
  }, [...fixture(name)]);
  await page.locator('#drop-zone').dispatchEvent('drop', { dataTransfer: transfer });
  await transfer.dispose();
  await expect(page.getByRole('status')).toContainText('Attempt validated');
}
async function urlLoad(page: Page, url: string): Promise<void> {
  await page.getByLabel('Recording URL', { exact: true }).fill(url);
  await page.getByRole('button', { name: 'Load URL', exact: true }).click();
}

test('stored recorder attempts replay initial/multiple frames, click, levels and terminal reset via drag/drop', async ({ page }, testInfo) => {
  await page.goto('./');
  await drop(page, 'recorder-baseline-0');
  await expect(page.locator('#game-id')).toHaveText('synthetic-recorder-v1');
  await expect(page.locator('#pixels')).toHaveText('  0   1\n  2   3');
  await expect(page.locator('#attempt-summary strong')).toHaveText(['2', '2', '1 / 3', '0.25 s']);
  await expect(page.locator('#attempt-identity')).toHaveText('Attempt 0 · Session 00000000000000000000000000000001');
  await expect(page.locator('#action-history button')).toHaveCount(2);
  await page.getByRole('button', { name: '#0 ACTION6 (1, 0)' }).click();
  await expect(page.locator('#observation-info')).toContainText('shape [2, 2, 2]');
  await expect(page.locator('#current-level')).toHaveText('Observed levels: 1/3');
  await expect(page.locator('#pixels')).toHaveText('  4   5\n  6   7');
  await page.locator('#frame-select').fill('1');
  await page.locator('#frame-select').dispatchEvent('change');
  await expect(page.locator('#pixels')).toHaveText('  8   9\n 10  11');
  await expect(page.locator('#prediction-info')).toContainText('Not recorded');
  await expect(page.locator('#metric-select')).toBeDisabled();
  await page.screenshot({ path: testInfo.outputPath('recorder-retro-english.png'), fullPage: true });
  await page.getByRole('button', { name: /^#1 ACTION1/ }).click();
  await expect(page.locator('#termination')).toHaveText('Closed: game_over');
  await expect(page.locator('#observation-info')).toContainText('GAME_OVER');
  await expect(page.locator('#action-history')).not.toContainText('RESET');
  // GAME_OVER was already finalized before the RESET; the RESET response is
  // the next file's initial observation, never an extra frame in the old file.
  await drop(page, 'recorder-baseline-1');
  await expect(page.locator('#attempt-identity')).toHaveText('Attempt 1 · Session 00000000000000000000000000000001');
  await expect(page.locator('#pixels')).toHaveText('  9   9\n  9   9');
  await expect(page.locator('#current-level')).toHaveText('Observed levels: 0/3');
  await expect(page.locator('#attempt-summary strong')).toHaveText(['1', '1', '0 / 3', '0.25 s']);
  await expect(page.locator('#action-history button')).toHaveCount(1);
  await expect(page.locator('#previous-canvas')).toBeHidden();
});

test('real cross-origin CORS server loads a stored prediction; HTTP/CORS failures recover to no-model replay', async ({ page }, testInfo) => {
  await page.goto('./');
  // No route interception: Chromium fetches actual HTTP bytes from port 4178.
  await urlLoad(page, 'http://127.0.0.1:4178/recorder-baseline-0.arc3');
  await expect(page.getByRole('status')).toContainText('Attempt validated');
  await expect(page.locator('#pixels')).toHaveText('  0   1\n  2   3');
  await expect(page.locator('#action-history button')).toHaveCount(2);
  await page.getByRole('button', { name: '#0 ACTION6 (1, 0)' }).click();
  await expect(page.locator('#timeline')).toHaveValue('1');
  await expect(page.locator('#current-level')).toHaveText('Observed levels: 1/3');
  await page.locator('#frame-select').fill('1');
  await page.locator('#frame-select').dispatchEvent('change');
  await expect(page.locator('#pixels')).toHaveText('  8   9\n 10  11');
  const response = page.waitForResponse('http://127.0.0.1:4178/recorder-prediction.arc3');
  await urlLoad(page, 'http://127.0.0.1:4178/recorder-prediction.arc3');
  expect((await response).headers()['access-control-allow-origin']).toBe('http://127.0.0.1:4177');
  await expect(page.getByRole('status')).toContainText('Attempt validated');
  await page.getByRole('button', { name: '#0 ACTION6 (1, 0)' }).click();
  await expect(page.locator('#prediction-canvas')).toBeVisible();
  await expect(page.locator('#difference-info')).toContainText('1 (4/4 pixels; 100.00%)');
  await expect(page.locator('#recorded-errors')).toContainText('MAE (palette-index units): 2');
  await expect(page.locator('#recorded-errors')).toContainText('MSE (squared palette-index units): 5');
  await expect(page.locator('#rationale-info')).toContainText('Synthetic prediction; not SDK output');
  await page.screenshot({ path: testInfo.outputPath('recorder-synthetic-prediction.png'), fullPage: true });
  await urlLoad(page, 'http://127.0.0.1:4178/missing.arc3');
  await expect(page.getByRole('alert')).toContainText('HTTP 404');
  await expect(page.locator('#result')).toBeHidden();
  await urlLoad(page, 'http://localhost:4177/.fixtures/recorder-baseline-0.arc3');
  await expect(page.getByRole('alert')).toContainText('CORS');
  await drop(page, 'recorder-baseline-0');
  await page.getByRole('button', { name: '#0 ACTION6 (1, 0)' }).click();
  await expect(page.locator('#prediction-canvas')).toBeHidden();
  await expect(page.locator('#difference-canvas')).toBeHidden();
  await expect(page.locator('#rationale-info')).toContainText('Not recorded');
  await expect(page.getByRole('alert')).toBeHidden();
});

test('small malformed/truncated/oversized/corrupt/unsupported files fail visibly and recover', async ({ page }) => {
  await page.goto('./');
  const good = fixture('recorder-baseline-0');
  const malformed = Buffer.from(good); malformed.write('ARX3', 0);
  const oversized = Buffer.from(good); oversized.writeBigUInt64LE(536870913n, 8);
  const corrupt = Buffer.from(good); corrupt[corrupt.length - 1]! ^= 1;
  const major = Buffer.from(good); major.writeUInt16LE(2, 4);
  for (const [name, buffer, message] of [
    ['malformed', malformed, /magic/i], ['truncated', good.subarray(0, -1), /incomplete/i],
    ['oversized', oversized, /length|limit/i], ['corrupt', corrupt, /checksum/i], ['major', major, /major/i],
  ] as const) {
    await page.locator('#file-input').setInputFiles({ name: `${name}.arc3`, mimeType: 'application/octet-stream', buffer });
    await expect(page.getByRole('alert')).toContainText(message);
    await expect(page.locator('#result')).toBeHidden();
  }
  await drop(page, 'recorder-baseline-1');
  await expect(page.getByRole('alert')).toBeHidden();
});
