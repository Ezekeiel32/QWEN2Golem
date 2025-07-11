'use server';
/**
 * @fileOverview A chat flow that interacts with a custom Aether Golem Python server.
 *
 * - ollamaChat - A function that sends prompts to the Golem server and returns responses.
 * - OllamaChatInput - The input type for the ollamaChat function.
 * - OllamaChatOutput - The return type for the ollamaChat function.
 */

import { z } from 'zod';

const OllamaChatInputSchema = z.object({
  prompt: z.string().describe('The prompt to send to the Ollama server.'),
  sessionId: z.string().optional().describe('The conversation session ID to maintain context.'),
  temperature: z.number().min(0).max(1).default(0.7).describe('The temperature to use for generating the response.'),
  fileContent: z.string().optional().describe('The text content of an uploaded file.'),
  golemActivated: z.boolean().optional(),
  activationPhrases: z.array(z.string()).optional().describe('The sacred phrases to activate the Golem.'),
  sefirotSettings: z.record(z.string(), z.number()).optional(),
});
export type OllamaChatInput = z.infer<typeof OllamaChatInputSchema>;

const OllamaChatOutputSchema = z.object({
  directResponse: z.string().describe("The direct, user-facing response from the Golem."),
  aetherAnalysis: z.string().optional().describe("The Golem's analysis of the mystical/quantum parameters."),
  recommendation: z.string().optional().describe("The Golem's practical recommendations."),
  golemStats: z.any().optional().describe('Detailed statistics from the Aether Golem consciousness core.'),
});
export type OllamaChatOutput = z.infer<typeof OllamaChatOutputSchema>;


export async function ollamaChat(input: OllamaChatInput): Promise<OllamaChatOutput> {
  const golemUrl = "https://0f2d286ce4b8.ngrok-free.app";
  
  // We construct the payload, passing all the golem control parameters from the UI.
  const { prompt, fileContent, ...restOfInput } = input;

  const payload = {
      ...restOfInput, // This will include sessionId, golemActivated, activationPhrases, etc.
      prompt,
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

    // This structure matches the robust parsing done on the Python side.
    return {
      directResponse: golemResponse.direct_response,
      aetherAnalysis: golemResponse.aether_analysis,
      recommendation: golemResponse.recommendation,
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
