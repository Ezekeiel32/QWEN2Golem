
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

// Environment variable for the Golem server URL
const GOLEM_SERVER_URL = process.env.GOLEM_SERVER_URL;

if (!GOLEM_SERVER_URL) {
  throw new Error(
    'GOLEM_SERVER_URL environment variable is not set. Please set it to your ngrok URL.'
  );
}

/**
 * Sends a validated request to the Golem server and returns the response.
 * @param input The data to send to the Golem server.
 * @returns The Golem server's response.
 */
export async function golemChat(input: GolemInput): Promise<GolemOutput> {
  // Validate the input against the Zod schema
  const validatedInput = GolemInputSchema.parse(input);

  const requestBody = {
    user_input: validatedInput.prompt,
    session_id: validatedInput.sessionId,
    temperature: validatedInput.temperature,
    file_content: validatedInput.fileContent,
    golem_activated: validatedInput.golemActivated,
    activation_phrases: validatedInput.activationPhrases,
    sefirot_settings: validatedInput.sefirotSettings,
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

    // Validate the output against the Zod schema
    const validatedOutput = GolemOutputSchema.parse(responseData);

    return validatedOutput;
  } catch (error) {
    console.error('Failed to fetch from Golem server:', error);
    if (error instanceof Error) {
      throw new Error(error.message);
    }
    throw new Error('An unknown error occurred while communicating with the Golem server.');
  }
}
