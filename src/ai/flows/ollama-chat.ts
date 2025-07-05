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
  context: z.array(z.string()).optional().describe('Conversation history to maintain context.'),
});
export type OllamaChatInput = z.infer<typeof OllamaChatInputSchema>;

const OllamaChatOutputSchema = z.object({
  response: z.string().describe('The response from the Ollama server.'),
  updatedContext: z.array(z.string()).optional().describe('The updated conversation history.'),
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
    const {prompt, context, temperature} = input;

    // Construct the prompt manually for a simple text generation task.
    // This is more robust for models that don't reliably produce structured JSON.
    const fullPrompt = `You are a helpful chatbot assistant. Use the context to answer user questions.

Context:
${(context || []).join('\n')}

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

    // Always maintain and update the context.
    const updatedContext = [
      ...(context || []), 
      `User: ${prompt}`, 
      `Assistant: ${responseText}`
    ];

    return {
      response: responseText,
      updatedContext: updatedContext,
    };
  }
);
