'use server';
/**
 * @fileOverview A chat flow that interacts with a custom Aether Golem Python server.
 *
 * - ollamaChat - A function that sends prompts to the Golem server and returns responses.
 * - OllamaChatInput - The input type for the ollamaChat function.
 * - OllamaChatOutput - The return type for the ollamaChat function.
 */

import { z } from 'genkit';

const OllamaHistoryItemSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string(),
});
export type OllamaHistoryItem = z.infer<typeof OllamaHistoryItemSchema>;

const OllamaChatInputSchema = z.object({
  prompt: z.string().describe('The prompt to send to the Ollama server.'),
  history: z.array(OllamaHistoryItemSchema).optional().describe('The conversation history.'),
  temperature: z.number().min(0).max(1).default(0.7).describe('The temperature to use for generating the response.'),
  fileContent: z.string().optional().describe('The text content of an uploaded file.'),
  golemActivated: z.boolean().optional(),
  shemPower: z.number().optional(),
  sefirotSettings: z.record(z.string(), z.number()).optional(),
});
export type OllamaChatInput = z.infer<typeof OllamaChatInputSchema>;

const OllamaChatOutputSchema = z.object({
  response: z.string().describe('The response from the Ollama server.'),
  golemStats: z.any().optional().describe('Statistics from the Aether Golem consciousness core.'),
});
export type OllamaChatOutput = z.infer<typeof OllamaChatOutputSchema>;


export async function ollamaChat(input: OllamaChatInput): Promise<OllamaChatOutput> {
  const golemUrl = process.env.GOLEM_SERVER_URL;
  if (!golemUrl) {
    throw new Error(
      'GOLEM_SERVER_URL is not set. Please add it to your .env file and point it to your Golem Python server ngrok URL.'
    );
  }
  
  // To support conversation history, we construct a single string prompt for the Golem server,
  // just like the previous flow did for Ollama.
  const { prompt, history = [], fileContent, ...restOfInput } = input;

  let fullPrompt = 'You are a helpful chatbot assistant. You must answer questions based on the provided conversation history and any attached file content.\n\n';

  history.forEach(message => {
      const role = message.role === 'user' ? 'User' : 'Assistant';
      fullPrompt += `${role}: ${message.content}\n`;
  });
  
  if (fileContent) {
    fullPrompt += `\n\nBase your answer on the following file content:\n\n---\n${fileContent}\n---`;
  }
  
  fullPrompt += `\n\nUser: ${prompt}\nAssistant:`;

  const payload = {
      ...restOfInput,
      prompt: fullPrompt, // Send the constructed full prompt
      temperature: input.temperature,
      golemActivated: input.golemActivated,
      shemPower: input.shemPower,
      sefirotSettings: input.sefirotSettings,
  };
  
  try {
    const response = await fetch(golemUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify(payload),
      cache: 'no-store',
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error("Golem server error response:", errorText);
        throw new Error(`Golem server returned a non-OK response: ${response.status} ${response.statusText}`);
    }

    const golemResponse = await response.json();

    if (golemResponse.error) {
        throw new Error(`Golem server returned an internal error: ${golemResponse.error}`);
    }

    return {
      response: golemResponse.response,
      golemStats: golemResponse, // Return the full response as stats
    };
  } catch (error) {
    console.error('Error calling Golem server:', error);
    if (error instanceof Error) {
        if (error.message.includes('ECONNREFUSED')) {
            throw new Error(`Connection to Golem server at ${golemUrl} was refused. Is your Python server and ngrok tunnel running?`);
        }
        throw new Error(`Failed to get response from Golem server: ${error.message}`);
    }
    throw new Error('An unknown error occurred while contacting the Golem server.');
  }
}
