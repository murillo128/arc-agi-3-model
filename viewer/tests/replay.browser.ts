import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

async function open(page: Page, name: string, navigate = true): Promise<void> {
  if (navigate) await page.goto('./');
  await page.locator('#file-input').setInputFiles({ name: `${name}.arc3`, mimeType: 'application/octet-stream', buffer: readFileSync(new URL(`../.fixtures/${name}.arc3`, import.meta.url)) });
  await expect(page.getByRole('status')).toContainText('Attempt validated');
}
async function jump(page: Page, position: number): Promise<void> {
  await page.locator('#observation-select').fill(String(position));
  await page.locator('#observation-select').dispatchEvent('change');
}
async function rgba(page: Page, id = 'observation-canvas'): Promise<number[]> {
  return page.locator(`#${id}`).evaluate((element: HTMLCanvasElement) => [...element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data]);
}
const initialRGBA = [255, 255, 255, 255, 204, 204, 204, 255, 153, 153, 153, 255, 102, 102, 102, 255];

test('zero-step attempt renders native dimensions and meaningful unavailable states', async ({ page }) => {
  await open(page, 'zero-step');
  await expect(page.locator('#observation-canvas')).toHaveAttribute('width', '2');
  await expect(page.locator('#observation-canvas')).toHaveAttribute('height', '2');
  expect(await rgba(page)).toEqual(initialRGBA);
  await expect(page.locator('#previous-canvas')).toBeHidden();
  await expect(page.locator('#history-empty')).toBeVisible();
  await expect(page.locator('#play')).toBeDisabled();
  await expect(page.locator('#timeline')).toBeDisabled();
  await expect(page.locator('#prediction-info')).toHaveText('Unavailable (not recorded for this position).');
});

test('empty step result never borrows the pre-action bitmap', async ({ page }) => {
  await open(page, 'empty-result'); await jump(page, 1);
  await expect(page.locator('#observation-canvas')).toBeHidden();
  await expect(page.locator('#canvas-message')).toHaveText('No frames were returned for this observation.');
  await expect(page.locator('#frame-select')).toBeDisabled();
  await expect(page.locator('#animate')).toBeDisabled();
  expect(await rgba(page, 'previous-canvas')).toEqual(initialRGBA);
});

test('one-step click, past action, and frame selection preserve actual pixels and diagnostics', async ({ page }) => {
  await open(page, 'multi-frame');
  await page.getByRole('button', { name: '#0 ACTION6 (63, 0)' }).click();
  await expect(page.locator('#timeline')).toHaveValue('1');
  await expect(page.locator('#observation-info')).toContainText('shape [3, 2, 2]');
  expect(await rgba(page)).toEqual([102, 102, 102, 255, 153, 153, 153, 255, 204, 204, 204, 255, 255, 255, 255, 255]);
  expect(await rgba(page, 'previous-canvas')).toEqual(initialRGBA);
  await expect(page.locator('#prediction-info')).toContainText('Uncertainty: 0.5');
  await expect(page.locator('#learning-info')).toContainText('π');
  await page.locator('#frame-select').fill('2'); await page.locator('#frame-select').dispatchEvent('change');
  expect(await rgba(page)).toEqual([0, 255, 255, 255, 0, 255, 255, 255, 255, 123, 204, 255, 255, 255, 255, 255]);
  await expect(page.locator('#canvas-message')).toContainText('Unknown palette indices: 16, 255');
  await page.getByRole('button', { name: '↺ Restart', exact: true }).click();
  await expect(page.locator('#prediction-info')).toContainText('Unavailable');
});

test('level/status markers, scrubbing and past action selection synchronize the actual result', async ({ page }) => {
  await open(page, 'replay-levels');
  await page.getByRole('button', { name: 'Jump to Step 2 · L2 · WIN', exact: true }).click();
  await expect(page.locator('#observation-info')).toContainText('WIN · levels 2/2');
  await expect(page.locator('#observation-canvas')).toHaveAttribute('width', '10');
  await page.locator('#timeline').fill('2');
  await expect(page.locator('#observation-select')).toHaveValue('2');
  await expect(page.locator('#observation-info')).toContainText('NOT_FINISHED · levels 1/2');
  await expect(page.locator('#current-level')).toHaveText('Observed levels: 1/2');
  await expect(page.locator('#action-history button[aria-current="true"]')).toContainText('#1 ACTION6 (9, 4)');
  await page.locator('#action-history button').first().click();
  await expect(page.locator('#timeline')).toHaveValue('1');
  await expect(page.locator('#observation-canvas')).toHaveAttribute('width', '12');
  await expect(page.locator('#position-label')).toHaveText('Step 0 result');
  await expect(page.locator('#previous-info')).toContainText('levels 0/2');
});

test('closing RESET and an unexecuted reset request do not fabricate returned frames', async ({ page }) => {
  await open(page, 'closing-reset');
  await page.getByRole('button', { name: 'RESET · closing boundary' }).click();
  await expect(page.locator('#timeline')).toHaveValue('2');
  await expect(page.locator('#position-label')).toHaveText('Closing RESET · pre-reset observation');
  await expect(page.locator('#observation-info')).toContainText('RESET result unavailable');
  await expect(page.locator('#frame-select')).toHaveValue('2');
  expect(await rgba(page)).toEqual(await rgba(page, 'previous-canvas'));
  await expect(page.locator('#animate')).toBeDisabled();
  await expect(page.locator('#boundary-info')).toContainText('returned observation belongs to the next attempt');
  await open(page, 'unexecuted-reset', false);
  await expect(page.locator('#action-history button')).toHaveCount(0);
  await expect(page.locator('#boundary-info')).toContainText('Reset requested; no executed RESET was recorded.');
});

test('frame playback pauses, respects speed, and never advances the action', async ({ page }) => {
  await open(page, 'multi-frame'); await jump(page, 1); await page.clock.install();
  await page.locator('#frame-speed').selectOption('2');
  await page.locator('#animate').click(); await page.clock.runFor(500);
  await expect(page.locator('#frame-select')).toHaveValue('1');
  await page.locator('#animate').click(); await page.clock.runFor(1000);
  await expect(page.locator('#frame-select')).toHaveValue('1');
  await page.locator('#frame-speed').selectOption('6'); await page.locator('#animate').click();
  await page.clock.runFor(350);
  await expect(page.locator('#frame-select')).toHaveValue('2');
  await expect(page.locator('#animate')).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('#timeline')).toHaveValue('1');
});

test('keyboard navigation, episode speed and playback visit every frame then stop', async ({ page }) => {
  await open(page, 'replay-levels'); await page.clock.install();
  await page.locator('.screen').focus(); await page.keyboard.press('ArrowRight');
  await expect(page.locator('#timeline')).toHaveValue('1');
  await page.keyboard.press('ArrowLeft'); await expect(page.locator('#timeline')).toHaveValue('0');
  await page.locator('#episode-speed').selectOption('2'); await page.locator('#frame-speed').selectOption('2');
  await page.locator('.screen').focus(); await page.keyboard.press('Space'); await page.clock.runFor(500);
  await expect(page.locator('#timeline')).toHaveValue('1'); await expect(page.locator('#frame-select')).toHaveValue('0');
  await page.clock.runFor(250); await expect(page.locator('#frame-select')).toHaveValue('1');
  await page.keyboard.press('Space'); await page.clock.runFor(1000);
  await expect(page.locator('#timeline')).toHaveValue('1');
  await page.keyboard.press('Space'); await page.clock.runFor(500);
  await expect(page.locator('#timeline')).toHaveValue('2');
  await page.clock.runFor(250); await expect(page.locator('#frame-select')).toHaveValue('1');
  await page.clock.runFor(250); await expect(page.locator('#frame-select')).toHaveValue('2');
  await page.clock.runFor(1000); await expect(page.locator('#timeline')).toHaveValue('3');
  await expect(page.locator('#play')).toHaveAttribute('aria-pressed', 'false');
  await page.keyboard.press('Home'); await expect(page.locator('#timeline')).toHaveValue('0');
  await page.keyboard.press('End'); await expect(page.locator('#timeline')).toHaveValue('3');
  await page.locator('#url-input').focus(); await page.keyboard.press('ArrowLeft');
  await expect(page.locator('#timeline')).toHaveValue('3');
});

test('opening another recording clears playback and switching to an empty sequence clears the bitmap', async ({ page }) => {
  await open(page, 'replay-levels'); await page.locator('#play').click();
  await open(page, 'empty', false);
  await expect(page.locator('#play')).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('#observation-canvas')).toBeHidden();
  await expect(page.locator('#observation-canvas')).toHaveAttribute('width', '1');
  await expect(page.locator('#canvas-message')).toContainText('No frames were returned');
});

for (const [name, width, height] of [['desktop', 1440, 1100], ['narrow', 390, 844]] as const) {
  test(`hand-reviewed ${name} replay layout`, async ({ page }) => {
    await page.setViewportSize({ width, height }); await open(page, 'replay-levels'); await jump(page, 2);
    await expect(page.locator('#result')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page).toHaveScreenshot(`replay-${name}.png`, { fullPage: true, animations: 'disabled', maxDiffPixels: 100 });
  });
}
