import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uuid
import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator

# --- OpenAI-Compatible Data Models ---
# These models define the structure that Cursor expects to send and receive.

class Model(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "user"
    permission: List[Dict[str, Any]] = []
    # Add tool support capabilities
    supports_tools: bool = True
    supports_function_calling: bool = True
    capabilities: Dict[str, Any] = {
        "function_calling": True,
        "tools": True,
        "streaming": True,
        "temperature": True
    }

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

# --- Models for Non-Streaming Response ---

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    # A bit of a hack for timestamp
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]

# --- Models for Streaming Response ---

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# --- FastAPI Application ---

app = FastAPI()

# --- Add CORS Middleware ---
# This allows the Cursor IDE (running on a different origin)
# to make requests to our adapter server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


GOLEM_SERVER_URL = "http://localhost:5000/generate"
MODEL_NAME = "aether_golem"

@app.get("/v1/models")
async def list_models():
    """
    This endpoint provides a list of available models to Cursor.
    It's required for Cursor to recognize our custom Golem model.
    """
    return ModelList(
        data=[
            Model(
                id=MODEL_NAME,
                supports_tools=True,
                supports_function_calling=True,
                capabilities={
                    "function_calling": True,
                    "tools": True,
                    "streaming": True,
                    "temperature": True,
                    "max_tokens": True,
                    "top_p": True,
                    "frequency_penalty": True,
                    "presence_penalty": True
                }
            )
        ]
    )

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Get details for a specific model.
    Cursor sometimes calls this to check model capabilities.
    """
    if model_id == MODEL_NAME:
        return Model(
            id=MODEL_NAME,
            supports_tools=True,
            supports_function_calling=True,
            capabilities={
                "function_calling": True,
                "tools": True,
                "streaming": True,
                "temperature": True,
                "max_tokens": True,
                "top_p": True,
                "frequency_penalty": True,
                "presence_penalty": True
            }
        )
    else:
        return {"error": "Model not found"}

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "service": "Aether Golem Adapter"}

@app.get("/v1/engines")
async def list_engines():
    """
    Legacy OpenAI engines endpoint for compatibility.
    """
    return {
        "data": [
            {
                "id": MODEL_NAME,
                "object": "engine",
                "owner": "user",
                "ready": True
            }
        ]
    }

async def stream_golem_response(golem_response_content: str, model: str) -> AsyncGenerator[str, None]:
    """
    Simulates a streaming response by breaking the Golem's full text response
    into word-by-word chunks, formatted as Server-Sent Events (SSEs).
    """
    # First, send a chunk with the role
    role_chunk = ChatCompletionStreamResponse(
        model=model,
        choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)]
    )
    yield f"data: {role_chunk.json()}\n\n"

    # Then, stream the content word by word
    words = golem_response_content.split(" ")
    for word in words:
        if not word:
            continue
        # Add a space before each word to reconstruct the sentence
        content_chunk = ChatCompletionStreamResponse(
            model=model,
            choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(content=f" {word}"), finish_reason=None)]
        )
        yield f"data: {content_chunk.json()}\n\n"
        time.sleep(0.05)  # Small delay to simulate typing

    # Finally, send the stop signal
    stop_chunk = ChatCompletionStreamResponse(
        model=model,
        choices=[ChatCompletionStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")]
    )
    yield f"data: {stop_chunk.json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    This endpoint mimics OpenAI's chat completions endpoint.
    It translates requests from Cursor to the Golem server and back.
    """
    print(f"Received request: {request.dict()}")

    # 1. Extract the user's prompt from the incoming request.
    # We'll just take the last message as the prompt.
    user_prompt = ""
    if request.messages:
        user_prompt = request.messages[-1].content
    
    if not user_prompt:
        return {"error": "No prompt found in the request."}

    # 2. Handle tools if provided
    tools_context = ""
    if request.tools:
        tools_context = f"\n\nAvailable tools: {json.dumps(request.tools, indent=2)}"
        user_prompt += tools_context
        print(f"Tools provided: {len(request.tools)} tools")
    
    if request.tool_choice:
        tool_choice_context = f"\nTool choice preference: {request.tool_choice}"
        user_prompt += tool_choice_context
        print(f"Tool choice: {request.tool_choice}")

    # 3. Construct the request for our Golem server.
    golem_payload = {
        "prompt": user_prompt,
        "sessionId": f"cursor-session-{uuid.uuid4()}",
        "temperature": request.temperature,
        "golemActivated": True, # We assume activation for this endpoint
        "activationPhrases": [],
        "sefirotSettings": {}
    }

    print(f"Sending to Golem: {golem_payload}")

    try:
        # 4. Send the request to the Golem server.
        response = requests.post(GOLEM_SERVER_URL, json=golem_payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        golem_data = response.json()
        print(f"Received from Golem: {golem_data}")

        # 5. Extract the response and format it for Cursor (OpenAI standard).
        direct_response = golem_data.get("direct_response", "No direct response found.")
        aether_analysis = golem_data.get("aether_analysis", "")

        # Combine the direct response with the aether analysis for a richer reply.
        full_content = f"{direct_response}\n\n--- Aether Analysis ---\n{aether_analysis}"

        # 6. Create the OpenAI-compatible response object.
        # If the client requested a stream, return a StreamingResponse.
        if request.stream:
            return StreamingResponse(
                stream_golem_response(full_content, request.model),
                media_type="text/event-stream"
            )

        # Otherwise, return a regular JSON response.
        chat_message = ChatMessage(role="assistant", content=full_content)
        choice = ChatCompletionChoice(index=0, message=chat_message)
        chat_response = ChatCompletionResponse(model=request.model, choices=[choice])

        return chat_response

    except requests.exceptions.RequestException as e:
        print(f"Error contacting Golem server: {e}")
        return {"error": f"Failed to connect to Golem server at {GOLEM_SERVER_URL}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An internal error occurred in the adapter."}

if __name__ == "__main__":
    # To run this adapter: uvicorn golem_cursor_adapter:app --reload --port 8001
    print("Starting Golem Cursor Adapter Server...")
    print("Run with: uvicorn home.chezy.golem_cursor_adapter:app --reload --port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001) 