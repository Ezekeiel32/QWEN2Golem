
'use server';

import { z } from 'zod';

// Define the schema for the input to the Golem server
const GolemInputSchema = z.object({
  prompt: z.string(),
  sessionId: z.string(),
  temperature: z.number().optional(),
  fileContent: z.string().optional(),
  golemActivated: z.boolean().optional(),
  activationPhrases: z.array(z.string()).optional(),
  sefirotSettings: z.record(z.string(), z.number()).optional(),
});

export type GolemInput = z.infer<typeof GolemInputSchema>;

// Define the expected output schema from the Golem server
const GolemOutputSchema = z.object({
  directResponse: z.string().optional(),
  aetherAnalysis: z.string().nullable().optional(),
  recommendation: z.string().nullable().optional(),
  // Use .any() for complex nested objects that we don't need to validate deeply here
  golem_state: z.any().optional(),
  quality_metrics: z.any().optional(),
  golem_analysis: z.any().optional(),
  aether_data: z.any().optional(),
  server_metadata: z.any().optional(),
  search_performed: z.boolean().optional(),
  search_query: z.string().optional(),
  search_results: z.string().optional(),
});

export type GolemOutput = z.infer<typeof GolemOutputSchema>;

// Use the public backend URL from environment variables
const GOLEM_SERVER_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

// No need for a throw new Error check here as NEXT_PUBLIC_BACKEND_URL will have a default value from next.config.mjs
// if not explicitly set in the deployment environment.

/**
 * Sends a validated request to the Golem server and returns the response.
 * @param input The data to send to the Golem server.
 * @returns The Golem server's response.
 */
export async function golemChat(input: GolemInput): Promise<GolemOutput> {
  // Validate the input against the Zod schema
  const validatedInput = GolemInputSchema.parse(input);

  const requestBody = {
    prompt: validatedInput.prompt,
    sessionId: validatedInput.sessionId,
    temperature: validatedInput.temperature,
    fileContent: validatedInput.fileContent,
    golemActivated: validatedInput.golemActivated,
    activationPhrases: validatedInput.activationPhrases,
    sefirotSettings: validatedInput.sefirotSettings,
  };

  try {
    const response = await fetch(`${GOLEM_SERVER_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error('Golem server error response:', errorBody);
      throw new Error(`Golem server returned a non-OK response: ${response.status} ${response.statusText}`);
    }

    const responseData = await response.json();

    // Map the backend response to the expected frontend format
    const mappedResponse = {
      directResponse: responseData.direct_response,
      aetherAnalysis: responseData.aether_analysis,
      recommendation: responseData.recommendation,
      golem_state: responseData.golem_state,
      quality_metrics: responseData.quality_metrics,
      golem_analysis: responseData.golem_analysis,
      aether_data: responseData.aether_data,
      server_metadata: responseData.server_metadata,
      search_performed: responseData.search_performed,
      search_query: responseData.search_query,
      search_results: responseData.search_results,
    };

    // Validate the output against the Zod schema
    const validatedOutput = GolemOutputSchema.parse(mappedResponse);

    return validatedOutput;
  } catch (error) {
    console.error('Failed to fetch from Golem server:', error);
    if (error instanceof Error) {
      throw new Error(error.message);
    }
    throw new Error('An unknown error occurred while communicating with the Golem server.');
  }
}
