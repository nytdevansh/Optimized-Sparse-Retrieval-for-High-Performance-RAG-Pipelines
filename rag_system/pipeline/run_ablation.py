#!/usr/bin/env python3
"""
Driver script to run ablation studies. Expects a base config and a list of ablation overrides.
"""
import yaml
import json
import copy
from pathlib import Path
from rag_system.pipeline.rag_research_pipeline import load_config
from rag_system.pipeline.evaluate_rag_pipeline import run_rag_experiment


def deep_update(base_dict, update_dict):
    """Recursively update nested dictionaries."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def generate_ablation_configs(base_config, ablation_specs):
    """Generate configuration variants for ablation study."""
    configs = []
    
    for spec in ablation_specs:
        # Create a deep copy of base config
        variant = copy.deepcopy(base_config)
        
        # Apply the ablation override
        deep_update(variant, spec['override'])
        
        # Update experiment name
        if 'experiments' in variant and variant['experiments']:
            for exp in variant['experiments']:
                exp['name'] = f"{exp.get('name', 'experiment')}_{spec['name']}"
        
        configs.append({
            'name': spec['name'],
            'config': variant,
            'description': spec.get('description', '')
        })
    
    return configs


def run_ablation(base_cfg_path, ablation_specs_path=None, custom_ablations=None):
    """Run ablation studies with specified configurations."""
    
    # Load base configuration
    base_config = load_config(base_cfg_path)
    out_dir = Path(base_config.get('output_dir', 'results'))
    ablation_dir = out_dir / 'ablations'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or use ablation specifications
    if ablation_specs_path:
        with open(ablation_specs_path, 'r') as f:
            ablation_specs = yaml.safe_load(f)
    elif custom_ablations:
        ablation_specs = custom_ablations
    else:
        # Default ablation examples
        ablation_specs = [
            {
                'name': 'baseline',
                'description': 'Original configuration as baseline',
                'override': {}
            },
            {
                'name': 'reduced_topk',
                'description': 'Reduce retrieval top-k to 10',
                'override': {
                    'experiments': [
                        {
                            'retriever': {
                                'params': {
                                    'top_k': 10
                                }
                            }
                        }
                    ]
                }
            },
            {
                'name': 'increased_topk', 
                'description': 'Increase retrieval top-k to 100',
                'override': {
                    'experiments': [
                        {
                            'retriever': {
                                'params': {
                                    'top_k': 100
                                }
                            }
                        }
                    ]
                }
            }
        ]
    
    # Generate ablation configurations
    ablation_configs = generate_ablation_configs(base_config, ablation_specs)
    
    print(f"Running {len(ablation_configs)} ablation experiments...")
    
    # Run each ablation
    ablation_results = {}
    
    for i, ablation in enumerate(ablation_configs, 1):
        print(f"\n{'='*50}")
        print(f"Ablation {i}/{len(ablation_configs)}: {ablation['name']}")
        print(f"Description: {ablation['description']}")
        print(f"{'='*50}")
        
        try:
            # Create ablation-specific output directory
            ablation_out_dir = ablation_dir / ablation['name']
            ablation_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Run experiments for this ablation
            exp_results = {}
            config = ablation['config']
            
            for exp in config.get('experiments', []):
                exp_name = exp.get('name', f'experiment_{len(exp_results)}')
                print(f"  Running experiment: {exp_name}")
                
                try:
                    result = run_rag_experiment(exp, config, ablation_out_dir)
                    exp_results[exp_name] = result
                    print(f"  ✓ {exp_name} completed")
                except Exception as e:
                    print(f"  ✗ {exp_name} failed: {e}")
                    exp_results[exp_name] = {'error': str(e)}
            
            ablation_results[ablation['name']] = {
                'description': ablation['description'],
                'experiments': exp_results,
                'successful': sum(1 for r in exp_results.values() if 'error' not in r),
                'failed': sum(1 for r in exp_results.values() if 'error' in r)
            }
            
            print(f"✓ Ablation {ablation['name']} completed")
            
        except Exception as e:
            print(f"✗ Ablation {ablation['name']} failed: {e}")
            ablation_results[ablation['name']] = {
                'description': ablation['description'],
                'error': str(e)
            }
    
    # Save ablation results summary
    summary = {
        'base_config': base_cfg_path,
        'total_ablations': len(ablation_configs),
        'results': ablation_results
    }
    
    summary_file = ablation_dir / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAblation study completed!")
    print(f"Results saved to: {summary_file}")
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG ablation studies")
    parser.add_argument('--base', type=str, default='paper_results.yaml',
                       help='Base configuration file')
    parser.add_argument('--ablations', type=str, 
                       help='Ablation specifications file (YAML)')
    parser.add_argument('--list-examples', action='store_true',
                       help='Show example ablation configurations')
    
    args = parser.parse_args()
    
    if args.list_examples:
        print("Example ablation configurations:")
        print("1. Retrieval top-k variations")
        print("2. Different retriever models")
        print("3. Reader model comparisons")
        print("4. Context length variations")
        print("\nCreate a YAML file with 'name', 'description', and 'override' fields")
        exit(0)
    
    run_ablation(args.base, args.ablations)
