'use server';
/**
 * @fileOverview A chat flow that interacts with a Qwen 2 7b model hosted on an Ollama server.
 *
 * - ollamaChat - A function that sends prompts to the Ollama server and returns responses.
 * - OllamaChatInput - The input type for the ollamaChat function.
 * - OllamaChatOutput - The return type for the ollamaChat function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const OllamaChatInputSchema = z.object({
  prompt: z.string().describe('The prompt to send to the Ollama server.'),
  temperature: z.number().min(0).max(1).default(0.7).describe('The temperature to use for generating the response.'),
});
export type OllamaChatInput = z.infer<typeof OllamaChatInputSchema>;

const OllamaChatOutputSchema = z.object({
  response: z.string().describe('The response from the Ollama server.'),
});
export type OllamaChatOutput = z.infer<typeof OllamaChatOutputSchema>;

export async function ollamaChat(input: OllamaChatInput): Promise<OllamaChatOutput> {
  return ollamaChatFlow(input);
}

const ollamaChatFlow = ai.defineFlow(
  {
    name: 'ollamaChatFlow',
    inputSchema: OllamaChatInputSchema,
    outputSchema: OllamaChatOutputSchema,
  },
  async input => {
    const {prompt, temperature} = input;

    // A simple prompt for a stateless chatbot.
    const fullPrompt = `You are a helpful chatbot assistant. Answer the following question.
User: ${prompt}
Assistant:`;
    
    const llmResponse = await ai.generate({
      model: 'ollama/qwen2:7b-custom',
      prompt: fullPrompt,
      config: {
        temperature,
      },
    });

    const responseText = llmResponse.text;

    return {
      response: responseText,
    };
  }
);
