import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

def plot_advanced_comparison(results_dict, difficulties=['easy', 'medium']):
    """Create 2x3 comparison panel for all difficulties"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Agent RL Ablation Study: Advanced Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['sr', 'cols', 'steps']
    titles = ['Success Rate (%)', 'Collisions', 'Episode Steps']
    styles = {'A': ('--', '#7f7f7f'), 'B': ('-.', '#1f77b4'), 'C': ('-', '#2ca02c'), 
              'D': ('-', '#ff7f0e'), 'E': ('-', '#d62728')}
    
    for diff_idx, difficulty in enumerate(difficulties):
        results = results_dict.get(difficulty, {})
        
        for metric_idx, (metric_key, title) in enumerate(zip(metrics, titles)):
            ax = axes[diff_idx, metric_idx]
            
            for variant, (ls, color) in styles.items():
                if variant in results:
                    curve = results[variant][metric_key]
                    xs = [pt[0] for pt in curve]
                    if metric_key == 'sr':
                        ys = [pt[1] * 100 for pt in curve]
                    else:
                        ys = [pt[1] for pt in curve]
                    ax.plot(xs, ys, linestyle=ls, color=color, linewidth=2.5, label=variant, marker='o')
            
            ax.set_xlabel('Episodes', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} ({difficulty.upper()})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if metric_idx == 0:
                ax.legend(loc='best', fontsize=9)
    
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/advanced_comparison.png', dpi=200, bbox_inches='tight')
    print("✅ Advanced comparison figure saved to: figures/advanced_comparison.png")
    plt.close()


def analyze_scalability(results_dict):
    """Analyze how variants scale across difficulties"""
    
    difficulties = list(results_dict.keys())
    variants = list(results_dict[difficulties[0]].keys()) if difficulties else []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Scalability Analysis Across Difficulties', fontsize=14, fontweight='bold')
    
    for metric_idx, (metric, ax_title) in enumerate([('sr', 'Success Rate'), ('cols', 'Collisions')]):
        ax = axes[metric_idx]
        
        for variant in variants:
            final_values = []
            for diff in sorted(difficulties):
                if variant in results_dict[diff]:
                    curve = results_dict[diff][variant][metric]
                    if curve:
                        final_val = curve[-1][1]
                        if metric == 'sr':
                            final_val *= 100
                        final_values.append(final_val)
            
            if final_values:
                ax.plot(sorted(difficulties), final_values, marker='o', label=variant, linewidth=2.5)
        
        ax.set_xlabel('Difficulty', fontsize=11)
        ax.set_ylabel(ax_title, fontsize=11)
        ax.set_title(f'Final {ax_title} by Difficulty', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/scalability_analysis.png', dpi=200, bbox_inches='tight')
    print("✅ Scalability analysis saved to: figures/scalability_analysis.png")
    plt.close()
