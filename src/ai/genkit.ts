import {genkit} from 'genkit';
import {ollama} from 'genkitx-ollama';

export const ai = genkit({
  plugins: [
    ollama({
      serverAddress: 'https://81af-2a06-c701-9364-c400-a6f1-4b98-216b-5b28.ngrok-free.app',
      client: {
        requestTimeout: 120000, // 2 minutes
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    }),
  ],
});
