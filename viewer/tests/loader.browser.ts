import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

const oneFrame = readFileSync(new URL('../.fixtures/one-frame.arc3', import.meta.url));
const multiFrame = readFileSync(new URL('../.fixtures/multi-frame.arc3', import.meta.url));
async function picked(page: Page, buffer = oneFrame, name = 'attempt.arc3'): Promise<void> {
  await page.locator('#file-input').setInputFiles({ name, mimeType: 'application/octet-stream', buffer });
}
async function urlLoad(page: Page, url: string): Promise<void> {
  await page.getByLabel('Recording URL', { exact: true }).fill(url);
  await page.getByRole('button', { name: 'Load URL', exact: true }).click();
}
async function validated(page: Page): Promise<void> {
  await expect(page.getByRole('status')).toContainText('Attempt validated');
  await expect(page.locator('#game-id')).toHaveText('fixture-v1');
}

test('file dialog validates the Python sample under a nested static path with no external requests', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('./');
  await expect(page.getByRole('status')).toHaveText('Ready to open a recording.');
  await picked(page);
  await validated(page);
  await expect(page.locator('#pixels')).toHaveText('  0   1\n  2   3');
  await expect(page.locator('#metadata')).toContainText('Unavailable (not recorded)');
  expect(requests.every(url => new URL(url).origin === 'http://127.0.0.1:4177')).toBe(true);
  expect(requests.some(url => url.endsWith('.wasm'))).toBe(true);
});

test('drag/drop preserves every frame and renders hostile metadata literally', async ({ page }) => {
  await page.goto('./');
  const transfer = await page.evaluateHandle((bytes: number[]) => {
    const data = new DataTransfer();
    data.items.add(new File([new Uint8Array(bytes)], 'multi.arc3', { type: 'application/octet-stream' }));
    return data;
  }, [...multiFrame]);
  await page.locator('#drop-zone').dispatchEvent('drop', { dataTransfer: transfer });
  await validated(page);
  await expect(page.locator('#metadata')).toContainText('<img src=x onerror=alert(1)>');
  await expect(page.locator('#result img, #result script')).toHaveCount(0);
  await page.locator('#observation-select').fill('1');
  await page.locator('#observation-select').dispatchEvent('change');
  await expect(page.locator('#observation-info')).toContainText('ACTION6 {"x":63,"y":0,"extra":-2}');
  await expect(page.locator('#pixels')).toHaveText('  3   2\n  1   0');
  await page.locator('#frame-select').fill('1');
  await page.locator('#frame-select').dispatchEvent('change');
  await expect(page.locator('#pixels')).toHaveText('  3   2\n  1   0');
  await page.locator('#frame-select').fill('2');
  await page.locator('#frame-select').dispatchEvent('change');
  await expect(page.locator('#pixels')).toHaveText('255  16\n  7   0');
});

test('URL success requires a user action and sends neither credentials nor file data', async ({ page, context }) => {
  let calls = 0;
  await context.addCookies([{ name: 'private', value: 'secret', domain: 'fixture.example', path: '/', secure: true }]);
  await page.route('https://fixture.example/attempt.arc3', async route => {
    calls++;
    const request = route.request();
    expect(request.method()).toBe('GET');
    expect(request.postData()).toBeNull();
    expect(request.headers()).not.toHaveProperty('cookie');
    expect(request.headers()).not.toHaveProperty('referer');
    await route.fulfill({ body: oneFrame, headers: { 'access-control-allow-origin': '*' } });
  });
  await page.goto('./');
  await page.getByLabel('Recording URL', { exact: true }).fill('https://fixture.example/attempt.arc3');
  expect(calls).toBe(0);
  await page.getByRole('button', { name: 'Load URL', exact: true }).click();
  await validated(page);
  await expect(page.locator('#metadata')).toContainText('URL · https://fixture.example/attempt.arc3');
  expect(calls).toBe(1);
});

test('explicit HTTPS deep-link fetches its recording', async ({ page }) => {
  await page.route('https://fixture.example/deep.arc3', route => route.fulfill({ body: oneFrame, headers: { 'access-control-allow-origin': '*' } }));
  await page.goto(`./?file=${encodeURIComponent('https://fixture.example/deep.arc3')}`);
  await validated(page);
});

test('actual cross-origin CORS refusal and mocked network/HTTP failures recover to a local file', async ({ page }) => {
  await page.goto('./');
  // The same static fixture server via a different origin provides no CORS header.
  await urlLoad(page, 'http://localhost:4177/.fixtures/one-frame.arc3');
  await expect(page.getByRole('alert')).toContainText('CORS');
  await page.route('https://fixture.example/offline', route => route.abort('internetdisconnected'));
  await urlLoad(page, 'https://fixture.example/offline');
  await expect(page.getByRole('alert')).toContainText('offline');
  await page.route('https://fixture.example/missing', route => route.fulfill({ status: 404, headers: { 'access-control-allow-origin': '*' } }));
  await urlLoad(page, 'https://fixture.example/missing');
  await expect(page.getByRole('alert')).toContainText('HTTP 404');
  await picked(page);
  await validated(page);
  await expect(page.getByRole('alert')).toBeHidden();
});

test('corrupt checksum, invalid URL and over-limit URL recover without reload', async ({ page }) => {
  await page.goto('./');
  const corrupt = Buffer.from(oneFrame);
  corrupt[corrupt.length - 1]! ^= 1;
  await picked(page, corrupt);
  await expect(page.getByRole('alert')).toContainText('checksum');
  await expect(page.locator('#result')).toBeHidden();
  await urlLoad(page, 'file:///private.arc3');
  await expect(page.getByRole('alert')).toContainText('HTTP(S)');
  await page.route('https://fixture.example/huge', route => route.fulfill({ body: oneFrame, headers: { 'content-length': '600000000', 'access-control-allow-origin': '*' } }));
  await urlLoad(page, 'https://fixture.example/huge');
  await expect(page.getByRole('alert')).toContainText('memory limit');
  await picked(page);
  await validated(page);
});

test('cancellation and a second file supersede a pending URL', async ({ page }) => {
  await page.route('https://fixture.example/slow', async route => {
    await new Promise<void>(resolve => page.once('close', () => resolve()));
    await route.abort().catch(() => undefined);
  });
  await page.goto('./');
  await urlLoad(page, 'https://fixture.example/slow');
  await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('cancelled');
  await urlLoad(page, 'https://fixture.example/slow');
  await picked(page);
  await validated(page);
});

test('empty sequence stays empty and invalid deep-links cause no remote fetch', async ({ page }) => {
  let remote = false;
  page.on('request', request => { if (request.url().startsWith('http://fixture.example')) remote = true; });
  await page.goto(`./?file=${encodeURIComponent('http://fixture.example/a')}`);
  await expect(page.getByRole('alert')).toContainText('HTTPS');
  expect(remote).toBe(false);
  await picked(page, readFileSync(new URL('../.fixtures/empty.arc3', import.meta.url)));
  await validated(page);
  await expect(page.locator('#pixels')).toContainText('No frames were returned');
  await expect(page.locator('#frame-select')).toBeDisabled();
});
