import {genkit} from 'genkit';
import {ollama} from 'genkitx-ollama';

export const ai = genkit({
  plugins: [
    ollama({
      serverAddress: 'https://68bf-2a02-6680-1162-2917-991d-2dc2-d730-4e.ngrok-free.app',
      client: {
        requestTimeout: 120000, // 2 minutes
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    }),
  ],
});
