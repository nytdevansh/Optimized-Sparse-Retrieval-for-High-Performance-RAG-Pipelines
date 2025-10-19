#!/usr/bin/env python3
"""
Main orchestrator for the RAG research pipeline.
Usage:
    python rag_research_pipeline.py --config paper_results.yaml
"""
import argparse
import yaml
import sys
from pathlib import Path
from rag_system.pipeline.evaluate_rag_pipeline import run_rag_experiment


def load_config(path):
    """Load configuration file with error handling."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        if not config:
            raise ValueError("Empty configuration file")
        
        if 'experiments' not in config:
            raise ValueError("Configuration must contain 'experiments' field")
            
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RAG Research Pipeline")
    parser.add_argument('--config', type=str, default='paper_results.yaml',
                       help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup output directory
    out_dir = Path(cfg.get('output_dir', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get experiments
    experiments = cfg.get('experiments', [])
    if not experiments:
        print("Warning: No experiments found in configuration")
        return
    
    print(f"Found {len(experiments)} experiments to run")
    
    # Run experiments
    results = {}
    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get('name', f'experiment_{i}')
        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{len(experiments)}: {exp_name}")
        print(f"{'='*60}")
        
        try:
            result = run_rag_experiment(exp, cfg, out_dir)
            results[exp_name] = result
            print(f"✓ Experiment {exp_name} completed successfully")
        except Exception as e:
            print(f"✗ Experiment {exp_name} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results[exp_name] = {'error': str(e)}
    
    # Save overall results
    overall_results = {
        'config_file': args.config,
        'total_experiments': len(experiments),
        'successful_experiments': sum(1 for r in results.values() if 'error' not in r),
        'failed_experiments': sum(1 for r in results.values() if 'error' in r),
        'results': results
    }
    
    overall_file = out_dir / 'overall_results.json'
    import json
    with open(overall_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Successful: {overall_results['successful_experiments']}")
    print(f"Failed: {overall_results['failed_experiments']}")
    print(f"Results saved to: {overall_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
