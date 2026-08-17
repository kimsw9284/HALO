# HGCAL Agent for Logistics & Operations (HALO)

**HALO** (**H**GCAL **A**gent for **L**ogistics and **O**perations) is an AI-powered agent and Retrieval-Augmented Generation (RAG) framework designed for the High Granularity Calorimeter upgrade for CMS detector at CERN.

The system provides information from technical documentation and by fetching real-time component tracking metadata directly from the HGCAL construction database.

---

## Environment Setup & Installation

Follow the steps below to create a clean virtual environment and install all required dependencies. This agentic setup currently ONLY works with Fermilab-hosted API keys available for Fermilab employees.

### 1. Create Working Directory & Virtual Environment

```bash
# Create and enter the working directory
mkdir litellm
cd litellm

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Verify pip version
pip --version
```


### 2. Install Dependencies
Install LiteLLM (with proxy support), Uvicorn, and additional packages required for PDF parsing and vector indexing:

```bash
# Install LiteLLM and Uvicorn
pip install "litellm[proxy]" uvicorn scikit-learn pypdf requests

# Verify installation
litellm --help
uvicorn --version
```

### 3. API Configuration
The model endpoint requires an authorized API key provided by the Fermilab office:
```bash
# Set your API key
export HOSTED_VLLM_API_KEY="your_api_key_here" #without quotes

# Verify the environment variable is set
echo $HOSTED_VLLM_API_KEY
```

### 4. Running and Testing the LiteLLM Proxy

Start the LiteLLM proxy server using ```config.yaml```:
```bash
litellm --config config.yaml
```

Expected terminal output:
```bash
INFO: Starting LiteLLM Proxy
INFO: Running on [http://0.0.0.0:4000](http://0.0.0.0:4000)
```

Leave this terminal running in the background. Open a second terminal window and run the following to test it:
```bash
# Navigate to project and activate virtual environment
cd ~/litellm
source venv/bin/activate

# Verify available models
curl http://localhost:4000/models
# Expected output: { "data": [ { "id": "llama3" } ] }

# Test text generation endpoint
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "user",
        "content": "Explain Linux in one paragraph."
      }
    ]
  }'
# Expected output: { "choices": [ { "message": { "content": "Linux is..." } } ] }
```

### 5. Running H.A.L.O.
```bash
python agent_demo.py
```

**Example Usage:**

Static document retrieval through RAG
> **User:** What is a cassette and what is Fermilab's role in cassette production?  
> **HALO:** *Queries local TF-IDF index across `docs/` and synthesizes an accurate response bounded strictly by the document context.*

Live database lookup through HGCAPI integration
> **User:** Tell me about part 320EH0QH0010012  
> **HALO:** *Detects the 15-character barcode, triggers a GET request to `https://hgcapi-cmsr.web.cern.ch/part/320EH0QH0010012/full`, and formats full part details (batch, manufacturer, QC records, version, user insertion history) into readable markdown.*