import {genkit} from 'genkit';
import {ollama} from 'genkitx-ollama';

export const ai = genkit({
  plugins: [
    ollama({
      serverAddress: 'https://7545-2a0d-6fc2-6800-6600-390b-5bea-4875-a81d.ngrok-free.app',
      client: {
        requestTimeout: 120000, // 2 minutes
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    }),
  ],
});
