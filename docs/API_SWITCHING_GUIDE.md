# API Switching Guide: Google Gemini ↔ LLaMA (Ollama)

## Overview

The system now supports **two LLM backends**:
1. **Google Gemini** (default) - Cloud-based, requires API key
2. **LLaMA via Ollama** - Local, requires Ollama installation

## Quick Start

### Using Google Gemini (Default)

No changes needed - system uses Google Gemini by default.

```bash
# Make sure GOOGLE_API_KEY is set
export GOOGLE_API_KEY="your-key-here"  # Linux/Mac
# or
$env:GOOGLE_API_KEY="your-key-here"    # Windows PowerShell

# Run any script
python tests/run_single_heterogeneous_test.py
```

### Switching to LLaMA (Ollama)

#### 1. Install Ollama

Download from: https://ollama.ai

```bash
# Verify installation
ollama --version
```

#### 2. Pull LLaMA model

```bash
# Recommended: llama3.2 (smaller, faster)
ollama pull llama3.2

# Alternative: llama3.1 (larger, more capable)
ollama pull llama3.1
```

#### 3. Set environment variable

```bash
# Linux/Mac
export ACTIVE_API="llama"

# Windows PowerShell
$env:ACTIVE_API="llama"

# Or add to .env file
echo "ACTIVE_API=llama" >> .env
```

#### 4. Run scripts

```bash
python tests/run_single_heterogeneous_test.py
```

## Configuration Options

### Method 1: Environment Variable (Recommended)

```bash
# Google Gemini
export ACTIVE_API="google"

# LLaMA Ollama
export ACTIVE_API="llama"
```

### Method 2: Edit `config/settings.py`

```python
# Change line:
ACTIVE_API = os.getenv("ACTIVE_API", "google")  # Default

# To:
ACTIVE_API = os.getenv("ACTIVE_API", "llama")   # LLaMA default
```

### Method 3: `.env` File

Add to `.env` file:
```
ACTIVE_API=llama
```

## Model Configuration

### Google Gemini Models

Edit `config/settings.py`:
```python
LLM_MODEL = "gemini-2.0-flash"      # Default (fast, good quality)
# LLM_MODEL = "gemini-1.5-pro"      # More capable, slower
# LLM_MODEL = "gemini-1.5-flash"    # Legacy
```

### LLaMA Models (Ollama)

Edit `agents/api_abstraction.py`:
```python
class LlamaAPI(PhilosopherAPI):
    def __init__(self, model: str = "llama3.2", ...):  # Default
        # Change to:
        # model = "llama3.1"    # Larger model
        # model = "mistral"     # Alternative
```

## Comparison

| Feature | Google Gemini | LLaMA (Ollama) |
|---------|---------------|----------------|
| **Setup** | API key only | Install Ollama + download model |
| **Speed** | Fast (cloud) | Depends on hardware |
| **Cost** | Free tier: 200 req/day | Free (local) |
| **Privacy** | Data sent to Google | Fully local |
| **Quality** | Very high | Good (model-dependent) |
| **Rate Limits** | Yes (15 req/min free) | No |

## Troubleshooting

### Google Gemini Issues

**Error: `429 Rate Limit`**
- Free tier: 15 requests/minute, 200/day
- Solution: Wait or upgrade to paid plan

**Error: `GOOGLE_API_KEY not found`**
```bash
export GOOGLE_API_KEY="your-key-here"
```

### LLaMA (Ollama) Issues

**Error: `Connection refused`**
- Ollama not running
- Solution: Start Ollama daemon:
```bash
ollama serve
```

**Error: `Model not found`**
- Model not downloaded
- Solution:
```bash
ollama pull llama3.2
```

**Slow performance**
- LLaMA requires significant compute (4-8GB RAM minimum)
- Try smaller model: `llama3.2` instead of `llama3.1`

## Testing API Switch

```python
# Test script
from agents.api_abstraction import get_philosopher_api

api = get_philosopher_api()
print(f"Active API: {api.get_model_name()}")

llm = api.get_llm()
response = llm.invoke("What is virtue ethics?")
print(response)
```

## Implementation Details

**Files modified:**
- `config/settings.py` - Added `ACTIVE_API` setting
- `agents/api_abstraction.py` - API abstraction layer
- `agents/philosopher_agents.py` - Uses abstraction instead of direct Google import

**Architecture:**
```
philosopher_agents.py
    ↓
api_abstraction.py (Factory)
    ↓
GoogleAPI | LlamaAPI
    ↓
ChatGoogleGenerativeAI | Ollama
```

## Best Practices

1. **Development**: Use Google Gemini (faster feedback)
2. **Privacy-sensitive**: Use LLaMA local
3. **Large experiments**: Use LLaMA (no rate limits)
4. **Production**: Use Google Gemini (more reliable)

## Future Extensions

To add more APIs (e.g., OpenAI, Anthropic):

1. Create new class in `api_abstraction.py`:
```python
class OpenAIAPI(PhilosopherAPI):
    def get_llm(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4", ...)
```

2. Update factory:
```python
def get_philosopher_api():
    if ACTIVE_API == "openai":
        return OpenAIAPI()
    # ...
```

3. Set `ACTIVE_API="openai"`
