'use server';

/**
 * @fileOverview An AI agent that maintains conversation history for context.
 *
 * - contextAwareConversation - A function that handles the conversation with context.
 * - ContextAwareConversationInput - The input type for the contextAwareConversation function.
 * - ContextAwareConversationOutput - The return type for the contextAwareConversation function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const ContextAwareConversationInputSchema = z.object({
  message: z.string().describe('The current message from the user.'),
  conversationHistory: z
    .array(z.object({role: z.enum(['user', 'assistant']), content: z.string()}))
    .optional()
    .describe('The history of the conversation.'),
});

export type ContextAwareConversationInput = z.infer<
  typeof ContextAwareConversationInputSchema
>;

const ContextAwareConversationOutputSchema = z.object({
  response: z.string().describe('The response from the AI model.'),
  updatedConversationHistory: z.array(z.object({role: z.enum(['user', 'assistant']), content: z.string()})).describe('The updated conversation history.'),
});

export type ContextAwareConversationOutput = z.infer<
  typeof ContextAwareConversationOutputSchema
>;

export async function contextAwareConversation(
  input: ContextAwareConversationInput
): Promise<ContextAwareConversationOutput> {
  return contextAwareConversationFlow(input);
}

const shouldMaintainContextTool = ai.defineTool({
  name: 'shouldMaintainContext',
  description: 'Determines whether to maintain the conversation history for context.',
  inputSchema: z.object({
    currentMessage: z.string().describe('The current message from the user.'),
    conversationHistory: z
      .array(z.object({role: z.enum(['user', 'assistant']), content: z.string()}))
      .optional()
      .describe('The history of the conversation.'),
  }),
  outputSchema: z.boolean().describe('True if the context should be maintained, false otherwise.'),
},
async (input) => {
  // Implement logic to decide whether to maintain context based on the input
  // For example, check if the conversation is ongoing or if the user is starting a new topic
  return input.conversationHistory !== undefined && input.conversationHistory.length > 0;
});


const prompt = ai.definePrompt({
  name: 'contextAwareConversationPrompt',
  input: {schema: ContextAwareConversationInputSchema},
  output: {schema: ContextAwareConversationOutputSchema},
  tools: [shouldMaintainContextTool],
  prompt: `You are a helpful chatbot. Respond to the user's message, taking into account the conversation history if it is relevant.

{% if tools.shouldMaintainContext.result === true %}
Conversation History:
{% each conversationHistory %}
{{this.role}}: {{this.content}}
{% endeach %}
{% endif %}

User Message: {{{message}}}

Response:`, // Ensure the prompt ends with "Response:"
});

const contextAwareConversationFlow = ai.defineFlow(
  {
    name: 'contextAwareConversationFlow',
    inputSchema: ContextAwareConversationInputSchema,
    outputSchema: ContextAwareConversationOutputSchema,
  },
  async input => {
    const maintainContext = await shouldMaintainContextTool({
      currentMessage: input.message,
      conversationHistory: input.conversationHistory,
    });

    const promptInput = {
      ...input,
    };

    const {output} = await prompt(promptInput);

    const updatedConversationHistory = [
      ...(input.conversationHistory || []),
      {role: 'user', content: input.message},
      {role: 'assistant', content: output!.response},
    ];

    return {
      response: output!.response,
      updatedConversationHistory: updatedConversationHistory,
    };
  }
);
