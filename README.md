# Few-Shot ECG Analysis with Vision-Language Models

This repository contains the code and experimental setup for the use of Large Vision-Language Models (VLMs) with In-Context Learning (ICL) for electrocardiogram (ECG) interpretation in data-scarce environments.

## Project Structure

```
icl-vlm/
├── config/                    # Experiment configurations
│   ├── abnormal/              # Abnormal beat detection configs
│   ├── brugada/               # Brugada syndrome detection configs
│   └── waves/                 # Wave identification configs
├── data/                      # Dataset folders
│   ├── abnormal/              # Abnormal beat detection data
│   ├── brugada/               # Brugada syndrome data
│   └── waves/                 # Wave identification data
├── prompts/                   # System and user prompts for different tasks
│   ├── abnormal/              # Prompts for abnormal detection
│   ├── brugada/               # Prompts for Brugada detection
│   └── waves/                 # Prompts for wave identification
└── src/                       # Source code
    ├── main.py                # Main execution script
    ├── llm.py                 # VLM interaction and prompt processing
    ├── data.py                # Data loading and few-shot sample selection
    ├── utils.py               # Utility functions
    ├── brugada_utils.py       # Brugada-specific utilities
    └── run_experiments.sh     # Batch experiment execution
```

## Key Features

- **Zero-shot and Few-shot Learning**: Support for both zero-shot inference and few-shot learning with 5 or 10 examples
- **Multiple ECG Tasks**: 
  - Abnormal beat detection
  - Brugada syndrome identification  
  - Wave peak identification (P, QRS, T waves)
- **Flexible Input Representations**: Support for single-beat, single-lead, and full 12-lead ECG representations
- **Advanced Prompting**: Sophisticated prompt engineering with medical knowledge integration
- **Multiple VLM Support**: Compatible with OpenAI GPT models and local models via vllm/LMStudio
- **Comprehensive Evaluation**: Detailed performance metrics and confidence scoring

## Data

The datasets used in this project are sourced from PhysioNet repositories.

### Abnormal Beat Detection
- **Source**: [MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **Description**: Models are required to classify beats as either representing a normal sinus rhythm or exhibiting patterns consistent with a Left Bundle Branch Block (LBBB).

### Wave Peak Identification
- **Source**: [MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **Description**: This task aims to assess VLMs capabilities in identifying the location of basic ECG components like QRS complexes and P/T waves. Data was sourced from normal beats.

### Brugada Syndrome Detection
- **Source**: [Brugada HUCA Dataset](https://www.physionet.org/content/brugada-huca/1.0.0/)
- **Description**: Dataset for identifying Brugada syndrome patterns in ECG signals.

### Dependencies

- Python 3.12+
- OpenAI API access (for GPT models)
- All required Python packages are listed in `pyproject.toml`

Create a venv and install dependencies:
```bash
uv sync
```

### API Setup

For OpenAI models, set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For local models, ensure vllm is running on `localhost:8000`.