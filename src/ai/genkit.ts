import {genkit} from 'genkit';
import {ollama} from 'genkitx-ollama';

export const ai = genkit({
  plugins: [
    ollama({
      serverAddress: 'https://b308-2a0d-6fc2-6800-6600-7a73-4abf-f3aa-67cb.ngrok-free.app',
      client: {
        requestTimeout: 120000, // 2 minutes
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    }),
  ],
});
