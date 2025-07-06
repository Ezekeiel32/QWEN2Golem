import {genkit} from 'genkit';
import {ollama} from 'genkitx-ollama';

export const ai = genkit({
  plugins: [
    ollama({
      client: {
        requestTimeout: 120000, // 2 minutes
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    }),
  ],
});
