# Optimized Sparse Retrieval for High-Performance RAG Pipelines

## Environment Setup

### Python Version
This project requires Python 3.12. We strongly recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Dataset Download and Loading
```bash
mkdir -p datasets
cd datasets
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
unzip fiqa.zip
cd ..
```
### Installation
```bash
# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Benchmarks

### Running FiQA Benchmarks
```bash
# Run the FiQA benchmark suite
python bench/fiqa_benchmark.py
```

The benchmark will:
1. Download the FiQA dataset if not present
2. Run benchmarks for all implemented methods:
   - BM25 (CPU-optimized)
   - DPR (Dense Passage Retrieval)
   - Contriever
   - SPLADE

Results will be saved in the `results/` directory:
- `fiqa_*_results.json`: Detailed per-query results
- `fiqa_*_summary.json`: Method summaries
- `fiqa_benchmark_report.md`: Comprehensive report
- `fiqa_results.csv`: Data for analysis

### Running the Optimized version(FiQA Dataset)

```bash
python -m rag_system.pipeline.rag_research_pipeline --config rag_system/configs/paper_results.yaml

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