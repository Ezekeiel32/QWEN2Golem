
'use server';
/**
 * @fileOverview A chat flow that interacts with a custom Aether Golem Python server.
 *
 * - ollamaChat - A function that sends prompts to the Golem server and returns responses.
 * - OllamaChatInput - The input type for the ollamaChat function.
 * - OllamaChatOutput - The return type for the ollamaChat function.
 */

import { z } from 'zod';

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
  activationPhrases: z.array(z.string()).optional().describe('The sacred phrases to activate the Golem.'),
  sefirotSettings: z.record(z.string(), z.number()).optional(),
});
export type OllamaChatInput = z.infer<typeof OllamaChatInputSchema>;

const OllamaChatOutputSchema = z.object({
  response: z.string().describe('The response from the Ollama server.'),
  golemStats: z.any().optional().describe('Statistics from the Aether Golem consciousness core.'),
});
export type OllamaChatOutput = z.infer<typeof OllamaChatOutputSchema>;


export async function ollamaChat(input: OllamaChatInput): Promise<OllamaChatOutput> {
  // TEMPORARY FIX: Hardcoding the URL to bypass environment variable issues.
  // When your ngrok URL changes, you will need to update it here.
  const golemUrl = "https://106513250230.ngrok-free.app";
  
  // We construct the payload, passing all the golem control parameters from the UI.
  const { prompt, history = [], fileContent, ...restOfInput } = input;

  const payload = {
      ...restOfInput, // This will include golemActivated, activationPhrases, sefirotSettings, etc.
      prompt,
      history, 
      fileContent,
      temperature: input.temperature,
  };
  
  try {
    // We append the /generate endpoint here to make the URL cleaner.
    const response = await fetch(`${golemUrl}/generate`, {
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
      // The python script returns the full response object which contains the stats we want.
      golemStats: golemResponse, 
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
