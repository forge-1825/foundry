

# Model Distillation Pipeline Web UI

<div align="center">

![Forge1825](frontend/public/ForgeFoundry.png)

[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A powerful web-based platform for AI model distillation by Forge1825**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 🚀 Overview

Model Distillation Pipeline is a comprehensive web application that streamlines the process of distilling large language models into smaller, more efficient versions. Built with modern technologies and designed for ease of use, it provides an intuitive interface for managing the entire distillation workflow.

**📢 BETA Notice**: This release provides the foundational infrastructure for model distillation. Full functionality requires additional setup of model containers and understanding of the distillation workflow. See [Current Status](#-current-status---beta) for details.

## Features

- **Script Management & Execution**: Execute individual Python scripts with configurable parameters, monitor script status, and support sequential execution of multiple scripts (pipeline mode).
- **Script Configuration**: Configure input/output paths, model parameters, and runtime options. Save and load configurations.
- **Real-time Monitoring**: Display log output in real-time via WebSockets, show system resource utilization, and track pipeline progress with visual indicators.
- **File System Integration**: Browse output directories and files, view generated JSON files in a structured format.
- **Results Analysis**: View run history, analyze performance metrics, and compare outputs.
- **Model Evaluation Hub**: Evaluate and compare model performance across different datasets and metrics.

## Architecture

### Frontend

- React.js (v18+)
- Tailwind CSS for styling
- Lucide React for iconography
- React Router for navigation
- Recharts for data visualization

### Backend

- FastAPI (Python) for REST API and WebSocket support
- Pydantic for data validation
- Python 3.10+ compatibility
- Script execution management system

### Communication

- REST API for configuration and commands
- WebSockets for real-time logs and status updates
- File system access for reading/writing data files

## 🚀 Quick Start

Get up and running in under 5 minutes!

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Docker Compose)
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU (optional, for accelerated processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/forge1825/model-distillation-pipeline.git
   cd model-distillation-pipeline
   ```

2. **Copy environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the UI**
   
   Open your browser and navigate to: **http://localhost:3456**

That's it! 🎉 The application is now running.

### Quick Test

To verify the infrastructure is working:
1. Navigate to the Dashboard
2. Check that the UI loads correctly
3. Explore the Script Configuration page
4. Note: Full pipeline execution requires model containers (not included)

**⚠️ Important**: This release provides the infrastructure. To run actual model distillation:
- You need to set up vLLM model containers separately
- See the documentation for model requirements
- Review script documentation to understand the workflow

For detailed setup instructions, see [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)

## Remote Deployment with GPU Acceleration

For deploying the frontend/backend locally while using remote GPU resources for model inference:

### Remote Server Setup (with GPU)

Start the vLLM model containers on your remote GPU server:

```bash
# Teacher Model (port 8000)
docker run -d --gpus all --name teacher_model -p 8000:8000 \
  -v /home/headquarters/my_vllm_models/Phi-3-mini-4k-instruct-AWQ:/models/teacher \
  -e TRUST_REMOTE_CODE=True \
  -e CUDA_LAUNCH_BLOCKING=1 \
  vllm/vllm-openai:latest \
  --model /models/teacher \
  --quantization awq_marlin \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.5 \
  --port 8000

# Student Model (port 8001)
docker run -d --gpus all --name student_model -p 8001:8001 \
  -v /home/headquarters/my_vllm_models/Phi-3-mini-4k-instruct-AWQ:/models/student \
  -e TRUST_REMOTE_CODE=True \
  -e CUDA_LAUNCH_BLOCKING=1 \
  vllm/vllm-openai:latest \
  --model /models/student \
  --quantization awq_marlin \
  --gpu-memory-utilization 0.6 \
  --max-model-len 1024 \
  --port 8001

# Distilled Model (port 8003)
docker run -d --gpus all --name distilled_model -p 8003:8003 \
  -v /home/headquarters/my_vllm_models/Phi-3-mini-4k-instruct-AWQ:/models/distilled \
  -e TRUST_REMOTE_CODE=True \
  -e CUDA_LAUNCH_BLOCKING=1 \
  vllm/vllm-openai:latest \
  --model /models/distilled \
  --quantization awq_marlin \
  --gpu-memory-utilization 0.6 \
  --max-model-len 512 \
  --port 8003
```

### Local Machine Setup

1. **Setup SSH tunnel** to forward model ports and Docker socket:
   ```bash
   ssh -L 2375:localhost:2375 -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8003:localhost:8003 user@remote-server-ip -N
   ```

2. **Start the application** using regular Docker Compose:
   ```bash
   docker-compose up -d
   ```

The application will now run locally while utilizing the remote GPU for model inference. This setup was tested with a 16GB RTX 4080 GPU.

### Development Setup

#### VS Code Setup

If you're using VS Code, you might see warnings about unknown at-rules (@tailwind, @apply) in the CSS files. These are expected and won't affect the application. To fix these warnings, you can install the "Tailwind CSS IntelliSense" extension:

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Tailwind CSS IntelliSense"
4. Install the extension by Tailwind Labs

This extension will provide proper syntax highlighting and autocomplete for Tailwind CSS directives.

#### Frontend

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

#### Backend

1. Navigate to the backend directory:

```bash
cd backend
```

2. Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:

```bash
poetry install
```

4. Start the FastAPI server:

```bash
poetry run uvicorn app.main:app --reload --port 7433
```

## Project Structure

```
model-distillation-ui/
├── frontend/                 # React frontend application
│   ├── public/               # Static files
│   └── src/                  # Source code
│       ├── components/       # React components
│       ├── contexts/         # React context providers
│       ├── hooks/            # Custom React hooks
│       ├── pages/            # Page components
│       └── services/         # API service modules
├── backend/                  # FastAPI backend application
│   ├── app/                  # Application code
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core functionality
│   │   ├── models/           # Pydantic models
│   │   └── services/         # Business logic services
│   └── pyproject.toml        # Python dependencies
├── scripts/                  # Python scripts for the pipeline
│   ├── manual_extractor.py   # Data extraction script
│   ├── data_enrichment.py    # Data enrichment script
│   ├── teacher_pair_generation.py # Teacher pair generation script
│   ├── distillation.py       # Distillation training script
│   └── evaluate_distilled.py # Model evaluation script
└── docker-compose.yml        # Docker Compose configuration
```

## Pipeline Steps

1. **Content Extraction & Enrichment (run_data_enrichment.bat)**
   - Extract content from PDF files and websites
   - Process the extracted data to clean text, extract entities, and summarize content
   - Output: Enriched JSON data ready for teacher pair generation

2. **Teacher Pair Generation (teacher_pair_generation_vllm_hierarchical.py)**
   - Query the Llama 3 teacher model to generate "soft" target outputs from enriched records
   - Create hierarchical context for better knowledge transfer
   - Output: JSON file with input-output pairs for distillation

3. **Distillation Training (distillation_vllm_faster_improved.py)**
   - Train the Phi-3 Mini student model using the teacher pairs
   - Use LoRA adapters for efficient fine-tuning
   - Output: Trained adapter weights

4. **Model Merging (merge_model.py)**
   - Merge the trained LoRA adapters with the base Phi-3 Mini model
   - Output: Complete merged model ready for deployment

5. **Student Self-Study (student_self_study_enhanced.py)**
   - Allow the student model to further learn from the data through self-directed exploration
   - Implement curiosity-driven learning for better exploration
   - Output: Enhanced student model with improved knowledge

6. **Model Evaluation (evaluation.py)**
   - Evaluate the distilled model against the teacher model
   - Compare performance across various metrics and datasets
   - Output: Evaluation results and improvement suggestions

## Model Evaluation Hub with OpenEvals Integration

The Model Evaluation Hub provides a comprehensive interface for viewing and comparing evaluation results between different models (teacher and student) across various datasets and metrics. It integrates with OpenEvals to provide standardized evaluation of model performance.

### Features

- **Results Overview**: Browse all evaluation runs with key metadata like dataset type, size, and timestamp
- **Detailed Metrics**: View detailed performance metrics for each model across different evaluators
- **Example Comparison**: Analyze specific examples with side-by-side comparison of model outputs and reference answers
- **Summary Statistics**: View aggregated statistics and performance differences between models
- **Improvement Suggestions**: Get AI-generated suggestions for improving the student model based on evaluation results
- **Run Configuration**: Configure and trigger new evaluation runs directly from the UI
- **Real-time Monitoring**: Track evaluation progress with live status updates and logs

### Post-Distillation Assessment

OpenEvals is primarily used for post-distillation assessment to compare:

1. **Newly Distilled Student Model**: The output of the current distillation run
2. **Teacher Model**: The original larger model that serves as the quality benchmark
3. **Previous Student Model**: The previously distilled version for regression testing
4. **Base Student Model**: The pre-distillation foundation model to measure distillation uplift

This assessment helps determine:
- If the new student model is better/worse than the previous one
- How close the student is to the teacher on key metrics
- Whether distillation significantly improved over the base model
- Specific areas where the student needs more work
- If the new student model should be promoted for use in the interactive learning loop

### Using the Model Evaluation Hub

#### Viewing Results

1. Navigate to the Model Evaluation page in the sidebar (or go to `/scripts/evaluation`)
2. Browse the list of available evaluation results
3. Click on a result to view detailed metrics and examples
4. Use the comparison tools to analyze model performance differences

#### Running New Evaluations

1. Click the "Start New Evaluation" button
2. Configure the evaluation:
   - Select models to evaluate
   - Choose datasets to use
   - Select evaluators to apply
   - Set the maximum number of examples
3. Click "Start Evaluation" to begin
4. Monitor progress in the status panel

### Evaluation Components

#### Datasets

Multiple dataset types are supported:
- **Error Suggestion**: Tests the model's ability to diagnose and suggest solutions for common errors
- **Command Extraction**: Evaluates the model's ability to extract or generate appropriate commands
- **Question Answering**: Assesses general knowledge and reasoning capabilities

#### Evaluators

Multiple evaluation methods are available:
- **String Match**: Simple lexical comparison between outputs and references
- **Embedding Similarity**: Semantic comparison using vector embeddings
- **LLM-as-Judge**: Using a capable LLM to evaluate responses based on correctness and relevance

### Adding New Evaluation Results

Evaluation results are stored in the `results` directory as JSON files. Each evaluation result should have:

- A main JSON file with the evaluation data (e.g., `sample_evaluation_20230615.json`)
- An optional summary JSON file with the same name plus `_summary` suffix (e.g., `sample_evaluation_20230615_summary.json`)

The evaluation results will automatically appear in the Model Evaluation Hub when the application is running.

### Documentation

For more detailed information, see the following documentation:

- [OpenEvals Integration Whitepaper](docs/OpenEvals_Integration_Whitepaper.md): Overview of how OpenEvals is integrated into the distillation pipeline
- [OpenEvals Technical Implementation](docs/OpenEvals_Technical_Implementation.md): Technical details on the implementation
- [Model Evaluation Hub User Guide](docs/Model_Evaluation_Hub_User_Guide.md): Detailed guide on using the Model Evaluation Hub

## 🔧 Current Status - BETA

This is a **BETA release** (v0.1.0-beta) with the following status:

### ✅ What's Working
- Web UI with real-time monitoring via WebSockets
- Docker containerization for easy deployment
- Model interaction and query interface
- Script execution framework with progress tracking
- Basic results visualization components

### ⚠️ Requires Setup
- Full pipeline execution (requires external vLLM model containers)
- Results visualization (requires execution data)
- Model evaluation (depends on trained models)

### 🚧 Known Limitations
- Test suite is not yet implemented
- Some error handling could be improved
- Performance optimization needed for large datasets
- Limited to specific model architectures (Llama 3, Phi-3)
- Multiple script versions without clear documentation on which to use
- No included model containers (must be set up separately)
- Pipeline workflow documentation needs improvement

### 🤝 Help Wanted
We're actively seeking contributors for:
- **Test Coverage** - Help us build a comprehensive test suite
- **Documentation** - Improve user guides and API docs
- **Performance** - Optimize pipeline execution
- **Model Support** - Add support for more model architectures

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help!

## 📄 License

This project is licensed under the MIT License with Attribution requirement - see the [LICENSE](LICENSE) file for details.

**Important**: Any use of this software requires attribution to Forge1825.

## 🏢 About Forge1825

Model Distillation Pipeline is proudly developed and maintained by Forge1825, committed to advancing AI accessibility through efficient model optimization.

---

<div align="center">
Made with ❤️ by Forge1825

**[Website](https://forge1825.com)** • **[GitHub](https://github.com/forge1825)** • **[Contact](mailto:hello@forge1825.com)**
</div>
