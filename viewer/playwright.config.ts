import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '*.browser.ts',
  use: {
    baseURL: 'http://127.0.0.1:4177/dist/',
    browserName: 'chromium',
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {},
  },
  webServer: {
    command: 'python3 -m http.server 4177 --bind 127.0.0.1 --directory .',
    url: 'http://127.0.0.1:4177/dist/',
    reuseExistingServer: false,
  },
});
