// src/ai/flows/ollama-chat.ts
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

const shouldMaintainContextTool = ai.defineTool({
  name: 'shouldMaintainContext',
  description: 'Determines whether to maintain the conversation history for context in subsequent turns. Use this tool if the user query requires context from previous turns.',
  inputSchema: z.object({
    userQuery: z.string().describe('The current user query.'),
  }),
  outputSchema: z.boolean().describe('Whether to maintain the conversation history.'),
}, async (input) => {
  // Basic logic to decide whether to maintain context.
  // More sophisticated logic can be added here based on the user query.
  const query = input.userQuery.toLowerCase();
  return !query.includes('new topic') && !query.includes('reset');
});

const ollamaChatPrompt = ai.definePrompt({
  name: 'ollamaChatPrompt',
  tools: [shouldMaintainContextTool],
  input: {schema: OllamaChatInputSchema},
  output: {schema: OllamaChatOutputSchema},
  prompt: `You are a helpful chatbot assistant. Use the context to answer user questions.

Context:
{{#each context}}
  {{this}}
{{/each}}

User: {{{prompt}}}

Assistant:`,
  model: 'ollama/qwen2:7b-custom',
});

const ollamaChatFlow = ai.defineFlow(
  {
    name: 'ollamaChatFlow',
    inputSchema: OllamaChatInputSchema,
    outputSchema: OllamaChatOutputSchema,
  },
  async input => {
    const {prompt, context, temperature} = input;
    const shouldMaintainContext = await shouldMaintainContextTool({
      userQuery: prompt,
    });

    const {output} = await ollamaChatPrompt(
      {
        prompt,
        temperature,
        context: shouldMaintainContext ? context : [],
      },
      {
        config: {
          temperature,
        },
      }
    );

    const updatedContext = shouldMaintainContext
      ? [...(context || []), `User: ${prompt}`, `Assistant: ${output?.response || ''}`]
      : [];

    return {
      response: output!.response,
      updatedContext: updatedContext,
    };
  }
);
