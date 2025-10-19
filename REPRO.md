# Lightning Retrieval Benchmark Reproduction Guide

## Environment Setup

### Python Version
This project requires Python 3.12. We strongly recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation
```bash
# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Benchmarks

### Quick Start (FiQA Dataset)
```bash
# Run FiQA benchmarks with all models
python bench/run_fiqa_models.py

# Run with specific model (e.g., BM25 only)
python bench/run_fiqa_models.py --method bm25

# Run with smaller dataset for quick testing
python bench/run_fiqa_models.py --subset-size 2000
```

### Full Benchmark Suite
```bash
# Run all benchmarks
python bench/run_all.py

# Run specific dataset benchmarks
python bench/run_all.py --config bench/configs/msmarco_bm25.yaml
```

## Reproducibility Notes

### Random Seeds
All randomized operations use fixed seeds:
- numpy: 42
- torch: 42
- python: 42

### Hardware Requirements
- CPU: Any x86_64 or ARM64 processor
- RAM: Minimum 16GB recommended (8GB for subset testing)
- Storage: 10GB free space for datasets and indices

### Expected Run Times
- FiQA full benchmark: ~30 minutes
- Quick test (2000 docs): ~2 minutes
- Full MS MARCO subset: ~2 hours

### Output Files
Results are saved in:
- `results/<dataset>_<method>_results.json`: Detailed per-query results
- `results/<dataset>_<method>_summary.json`: Aggregated metrics
- `results/fiqa_comparison.md`: Human-readable comparison table