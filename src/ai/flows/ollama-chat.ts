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
    try {
      const {prompt, temperature, fileContent, history = []} = input;

      let fullPrompt = 'You are a helpful chatbot assistant. You must answer questions based on the provided conversation history and any attached file content.\n\n';

      // Format the conversation history into the prompt string.
      history.forEach(message => {
          const role = message.role === 'user' ? 'User' : 'Assistant';
          fullPrompt += `${role}: ${message.content}\n`;
      });
      
      // Add the current user prompt.
      let currentPromptContent = prompt;
      if (fileContent) {
        fullPrompt += `\n\nBase your answer on the following file content:\n\n---\n${fileContent}\n---`;
      }
      
      fullPrompt += `\n\nUser: ${currentPromptContent}\nAssistant:`;

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
    } catch (error) {
        console.error("Error in ollamaChatFlow:", error);
        if (error instanceof Error) {
            throw new Error(`AI model request failed: ${error.message}`);
        }
        throw new Error('An unknown error occurred in the AI flow.');
    }
  }
);
