import { decodeAttempt } from './decoder';

self.onmessage = async (event: MessageEvent<Uint8Array>) => {
  try {
    const value = await decodeAttempt(event.data);
    // Every tensor BIN is a view into the same MessagePack output buffer.
    const buffer = value.attempt.initial_observation.frames.data.buffer;
    self.postMessage({ value }, { transfer: [buffer] });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
