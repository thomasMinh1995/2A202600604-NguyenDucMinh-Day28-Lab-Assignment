from fastapi import FastAPI, Request
import uvicorn
import threading
import random

# App 1: vLLM Mock Server (Port 8081)
vllm_app = FastAPI(title="Mock vLLM Server")

@vllm_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages[-1]["content"] if messages else ""
    
    # Return mock responses depending on query
    content = f"This is a mock AI response to your query: '{prompt}'. Received prompt with context details successfully."
    
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }
        ],
        "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    }

@vllm_app.get("/health")
async def health():
    return {"status": "ok"}

# App 2: Embedding Mock Server (Port 8082)
embed_app = FastAPI(title="Mock Embedding Server")

@embed_app.post("/embed")
async def embed(request: Request):
    body = await request.json()
    texts = body.get("texts", [])
    
    # Generate mock embeddings of size 384
    embeddings = []
    for text in texts:
        # Seed generator based on text for reproducibility
        random.seed(hash(text))
        emb = [random.uniform(-0.1, 0.1) for _ in range(384)]
        embeddings.append(emb)
        
    return {"embeddings": embeddings}

def run_vllm():
    uvicorn.run(vllm_app, host="0.0.0.0", port=8081, log_level="warning")

def run_embed():
    uvicorn.run(embed_app, host="0.0.0.0", port=8082, log_level="warning")

if __name__ == "__main__":
    print("Starting mock vLLM server on port 8081...")
    t1 = threading.Thread(target=run_vllm, daemon=True)
    t1.start()
    
    print("Starting mock Embedding server on port 8082...")
    t2 = threading.Thread(target=run_embed, daemon=True)
    t2.start()
    
    # Keep main thread alive
    t1.join()
    t2.join()
