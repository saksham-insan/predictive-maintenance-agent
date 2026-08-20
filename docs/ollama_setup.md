# Ollama & Local LLM Setup Guide

This guide provides instructions for configuring and running the local LLM backend (`llama3.2`) for the **Predictive Maintenance Autonomous Agent**.

---

## 1. Quick Start: Install & Run Ollama

### Step 1: Install Ollama
- **Windows**: Download and run the installer from [ollama.com/download/windows](https://ollama.com/download/windows).
- **macOS**: Download from [ollama.com/download/mac](https://ollama.com/download/mac) or install via Homebrew (`brew install ollama`).
- **Linux**: Run:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### Step 2: Pull the Recommended Model
The default configuration uses Meta's lightweight, high-performance `llama3.2` (3B parameters), which runs efficiently on local CPU or GPU:
```bash
ollama pull llama3.2
```

### Step 3: Verify Ollama Service
Ensure Ollama is running and responding on its default port:
```bash
# In PowerShell / Bash:
curl http://localhost:11434/api/tags
```
You should receive a JSON response listing `llama3.2`.

---

## 2. Environment Variables & Configuration

The agentic pipeline automatically detects and switches between backends based on environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LLM_BACKEND` | `ollama` | Selected backend: `ollama`, `mock`, or `groq` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama REST API host URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name to invoke (e.g. `llama3.2`, `mistral`, `qwen2.5`) |
| `GROQ_API_KEY` | *(None)* | Optional Groq API Key if using `LLM_BACKEND=groq` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq cloud model |

### Setting Environment Variables (PowerShell)
```powershell
# To use local Ollama:
$env:LLM_BACKEND="ollama"
$env:OLLAMA_MODEL="llama3.2"

# To use offline Mock agent for automated testing / CI:
$env:LLM_BACKEND="mock"

# To use free-tier Groq API:
$env:LLM_BACKEND="groq"
$env:GROQ_API_KEY="gsk_your_groq_api_key_here"
```

---

## 3. Testing the Agentic Layer

### Run the Agentic Pipeline from Terminal
```bash
# Run standalone agentic pipeline test:
python src/agents/agentic_pipeline.py
```

### Run the Interactive Live Dashboard
```bash
streamlit run dashboard/app.py
```
Inside the dashboard:
1. In the sidebar under **Pipeline Architecture**, select **Agentic AI (LLM + ReAct + RAG)**.
2. Select your active **Backend Provider** (Ollama, Mock Agent, or Groq).
3. Click **▶ Start simulation** or enter manual sensor telemetry.

---

## 4. Troubleshooting & FAQ

### Issue: `ConnectionError: Could not connect to Ollama at 'http://localhost:11434'`
- **Cause**: Ollama background service is not started.
- **Fix**: Open a terminal and run `ollama serve`, or start the Ollama desktop application from the Windows Start menu.

### Issue: `Model 'llama3.2' not found`
- **Fix**: Run `ollama pull llama3.2` in terminal to download model weights.

### Air-Gapped / Offline Environments
If you are developing or testing in an environment without internet access or GPU acceleration, simply set:
```bash
export LLM_BACKEND=mock   # Linux/macOS
$env:LLM_BACKEND="mock"    # Windows PowerShell
```
The `MockLLMClient` provides deterministic multi-step ReAct reasoning traces grounded in physics formulas for all failure modes (OSF, TWF, HDF, PWF, RNF).
