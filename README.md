

# Model Distillation Pipeline Web UI

A web-based user interface for managing a model distillation pipeline. This application enables users to execute Python scripts in a logical sequence to perform AI model distillation tasks, from data extraction to model evaluation.

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

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js (for local development)
- Python 3.10+ (for local development)

### Installation

1. Install dependencies:

```bash
# Run the install_dependencies.bat script
install_dependencies.bat
```

Or manually install dependencies:

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
cd frontend
npm install --legacy-peer-deps
cd ..
```

2. Set up environment variables:

A `.env` file has already been created in the root directory with default values. You can modify it if needed:

```
DATA_DIR=/path/to/your/data/directory
```

3. Build and start the Docker containers:

```bash
# Make sure you're in the model-distillation-ui directory

# Start the containers
docker-compose up --build
```

4. Alternatively, you can run the backend and frontend separately:

```bash
# Start the backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 7433 --reload

# In another terminal, start the frontend
cd frontend
npm start
```

5. Access the web UI at http://localhost:3456

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
