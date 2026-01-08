
# LUMIN v2: Neurosymbolic Query Engine for NASA PDS

This repository contains the core logic for LUMIN v2, a neurosymbolic search agent designed to translate abstract scientific queries into precise schema constraints for the Planetary Data System (PDS).

## Prerequisites

* Python 3.10 or higher
* Git
* An OpenAI API Key (for the Agentic component)

## Setup Instructions

### 1. Clone the Repository
Open your terminal and run the following commands to download the codebase.

```bash
git clone 
cd lumin-v2-core

```

### 2. Set Up Virtual Environment

Always run this project inside a virtual environment to manage dependencies.

**For macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

**For Windows (PowerShell):**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

Install the required Python packages.

```bash
pip install openai pydantic

```

*(Note: If a requirements.txt file is present, run `pip install -r requirements.txt` instead)*

### 4. Configure Environment Variables

The agent requires an API key to function.

1. Create a file named `.env` in the root directory (same level as this README).
2. Add your API key to the file:

```text
OPENAI_API_KEY=sk-your-key-here

```

3. **Important:** Never commit your `.env` file to GitHub. It is already excluded in `.gitignore`.

## Running the System

To verify your installation, run the logic engine demo. This tests the core neurosymbolic intersection logic (including cyclic time handling).

```bash
python run_demo.py

```

To run the full agentic loop (requires API key):

```bash
python -m src.agent

```

## Project Structure

* `src/logic_engine.py`: The core polymorphic solver that handles set operations and math.
* `src/agent.py`: The LLM entry point that translates natural language into logic plans.
* `data/ontology.json`: The "Ground Truth" definition file derived from PDS4 standards.
* `run_demo.py`: Integration test script for validting logic operations.

```

```