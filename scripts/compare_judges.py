#!/usr/bin/env python3
"""
Judge Comparison Script

This script runs both rule-based and OpenAI evaluation on the same test results
and provides a detailed comparison of their outputs and performance.

Author: AI Testing System
Date: 2025-09-27
"""

import json
import argparse
import logging
import time
import os
from typing import List, Dict, Any
import statistics
from dataclasses import asdict

# Import our judges
from judge import ResponseEvaluator, load_test_results as load_test_results_rule, load_paper_content as load_paper_content_rule
from openai_judge import OpenAIResponseEvaluator, load_test_results as load_test_results_openai, load_paper_content as load_paper_content_openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_evaluations(rule_results: List, openai_results: List) -> Dict[str, Any]:
    """
    Compare rule-based and OpenAI evaluation results.
    
    Args:
        rule_results (List): Results from rule-based evaluator
        openai_results (List): Results from OpenAI evaluator
        
    Returns:
        Dict: Comprehensive comparison analysis
    """
    comparison = {
        "summary": {},
        "metric_differences": {},
        "agreement_analysis": {},
        "cost_analysis": {},
        "performance_analysis": {}
    }
    
    # Create lookup for matching results
    rule_lookup = {r.id if hasattr(r, 'id') else r['id']: r for r in rule_results}
    openai_lookup = {r.id if hasattr(r, 'id') else r['id']: r for r in openai_results}
    
    # Find common test IDs
    common_ids = set(rule_lookup.keys()) & set(openai_lookup.keys())
    logger.info(f"Comparing {len(common_ids)} common test results")
    
    if not common_ids:
        logger.error("No common test IDs found between rule-based and OpenAI results")
        return comparison
    
    # Calculate metric differences
    relevancy_diffs = []
    adequacy_diffs = []
    clarity_diffs = []
    consistency_diffs = []
    
    rule_scores = {"relevancy": [], "adequacy": [], "clarity": [], "consistency": []}
    openai_scores = {"relevancy": [], "adequacy": [], "clarity": [], "consistency": []}
    
    total_openai_cost = 0.0
    total_openai_time = 0.0
    
    for test_id in common_ids:
        rule_result = rule_lookup[test_id]
        openai_result = openai_lookup[test_id]
        
        # Extract scores (handle both dataclass and dict formats)
        if hasattr(rule_result, 'relevancy'):
            r_rel, r_adeq, r_clar, r_cons = rule_result.relevancy, rule_result.adequacy, rule_result.clarity, rule_result.confusion
        else:
            r_rel, r_adeq, r_clar, r_cons = rule_result['relevancy'], rule_result['adequacy'], rule_result['clarity'], rule_result['confusion']
        
        if hasattr(openai_result, 'relevancy'):
            o_rel, o_adeq, o_clar, o_cons = openai_result.relevancy, openai_result.adequacy, openai_result.clarity, openai_result.confusion
            total_openai_cost += openai_result.evaluation_cost
            total_openai_time += openai_result.evaluation_time
        else:
            o_rel, o_adeq, o_clar, o_cons = openai_result['relevancy'], openai_result['adequacy'], openai_result['clarity'], openai_result['confusion']
            total_openai_cost += openai_result.get('evaluation_cost', 0)
            total_openai_time += openai_result.get('evaluation_time', 0)
        
        # Calculate differences
        relevancy_diffs.append(o_rel - r_rel)
        adequacy_diffs.append(o_adeq - r_adeq)
        clarity_diffs.append(o_clar - r_clar)
        consistency_diffs.append(o_cons - r_cons)
        
        # Store scores for analysis
        rule_scores["relevancy"].append(r_rel)
        rule_scores["adequacy"].append(r_adeq)
        rule_scores["clarity"].append(r_clar)
        rule_scores["consistency"].append(r_cons)
        
        openai_scores["relevancy"].append(o_rel)
        openai_scores["adequacy"].append(o_adeq)
        openai_scores["clarity"].append(o_clar)
        openai_scores["consistency"].append(o_cons)
    
    # Summary statistics
    comparison["summary"] = {
        "total_comparisons": len(common_ids),
        "rule_avg_scores": {
            metric: statistics.mean(scores) for metric, scores in rule_scores.items()
        },
        "openai_avg_scores": {
            metric: statistics.mean(scores) for metric, scores in openai_scores.items()
        }
    }
    
    # Metric differences analysis
    comparison["metric_differences"] = {
        "relevancy": {
            "mean_diff": statistics.mean(relevancy_diffs),
            "median_diff": statistics.median(relevancy_diffs),
            "std_diff": statistics.stdev(relevancy_diffs) if len(relevancy_diffs) > 1 else 0,
            "max_diff": max(relevancy_diffs),
            "min_diff": min(relevancy_diffs)
        },
        "adequacy": {
            "mean_diff": statistics.mean(adequacy_diffs),
            "median_diff": statistics.median(adequacy_diffs),
            "std_diff": statistics.stdev(adequacy_diffs) if len(adequacy_diffs) > 1 else 0,
            "max_diff": max(adequacy_diffs),
            "min_diff": min(adequacy_diffs)
        },
        "clarity": {
            "mean_diff": statistics.mean(clarity_diffs),
            "median_diff": statistics.median(clarity_diffs),
            "std_diff": statistics.stdev(clarity_diffs) if len(clarity_diffs) > 1 else 0,
            "max_diff": max(clarity_diffs),
            "min_diff": min(clarity_diffs)
        },
        "consistency": {
            "mean_diff": statistics.mean(consistency_diffs),
            "median_diff": statistics.median(consistency_diffs),
            "std_diff": statistics.stdev(consistency_diffs) if len(consistency_diffs) > 1 else 0,
            "max_diff": max(consistency_diffs),
            "min_diff": min(consistency_diffs)
        }
    }
    
    # Agreement analysis
    exact_matches = 0
    close_matches = 0  # Within 1 point
    
    for test_id in common_ids:
        rule_result = rule_lookup[test_id]
        openai_result = openai_lookup[test_id]
        
        # Extract scores
        if hasattr(rule_result, 'relevancy'):
            rule_total = rule_result.relevancy + rule_result.adequacy + rule_result.clarity + rule_result.confusion
        else:
            rule_total = rule_result['relevancy'] + rule_result['adequacy'] + rule_result['clarity'] + rule_result['confusion']
        
        if hasattr(openai_result, 'relevancy'):
            openai_total = openai_result.relevancy + openai_result.adequacy + openai_result.clarity + openai_result.confusion
        else:
            openai_total = openai_result['relevancy'] + openai_result['adequacy'] + openai_result['clarity'] + openai_result['confusion']
        
        if abs(rule_total - openai_total) == 0:
            exact_matches += 1
        elif abs(rule_total - openai_total) <= 4:  # Within 1 point per metric
            close_matches += 1
    
    comparison["agreement_analysis"] = {
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(common_ids),
        "close_matches": close_matches,
        "close_match_rate": (exact_matches + close_matches) / len(common_ids),
        "correlation_analysis": "Calculated based on total score differences"
    }
    
    # Cost analysis
    comparison["cost_analysis"] = {
        "total_openai_cost": total_openai_cost,
        "avg_cost_per_evaluation": total_openai_cost / len(common_ids) if common_ids else 0,
        "rule_based_cost": 0.0,
        "cost_savings_rule_based": total_openai_cost
    }
    
    # Performance analysis
    comparison["performance_analysis"] = {
        "total_openai_time": total_openai_time,
        "avg_openai_time_per_evaluation": total_openai_time / len(common_ids) if common_ids else 0,
        "estimated_rule_time": len(common_ids) * 0.01,  # Very fast
        "speed_advantage_rule_based": f"{total_openai_time / (len(common_ids) * 0.01):.1f}x faster"
    }
    
    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("🔬 RULE-BASED vs OPENAI JUDGE COMPARISON REPORT")
    print("="*80)
    
    summary = comparison["summary"]
    print(f"\n📊 SUMMARY")
    print(f"Total Comparisons: {summary['total_comparisons']}")
    
    print(f"\n🤖 Rule-Based Average Scores:")
    for metric, score in summary["rule_avg_scores"].items():
        print(f"   {metric.capitalize()}: {score:.2f}/5")
    
    print(f"\n🧠 OpenAI Average Scores:")
    for metric, score in summary["openai_avg_scores"].items():
        print(f"   {metric.capitalize()}: {score:.2f}/5")
    
    print(f"\n📈 METRIC DIFFERENCES (OpenAI - Rule-Based)")
    diffs = comparison["metric_differences"]
    for metric, diff_stats in diffs.items():
        print(f"\n{metric.capitalize()}:")
        print(f"   Mean Difference: {diff_stats['mean_diff']:+.2f}")
        print(f"   Range: {diff_stats['min_diff']:+.1f} to {diff_stats['max_diff']:+.1f}")
        print(f"   Std Deviation: {diff_stats['std_diff']:.2f}")
        
        if diff_stats['mean_diff'] > 0.5:
            print(f"   👆 OpenAI scores significantly HIGHER")
        elif diff_stats['mean_diff'] < -0.5:
            print(f"   👇 OpenAI scores significantly LOWER")
        else:
            print(f"   ⚖️  Scores relatively similar")
    
    print(f"\n🤝 AGREEMENT ANALYSIS")
    agreement = comparison["agreement_analysis"]
    print(f"Exact Matches: {agreement['exact_matches']} ({agreement['exact_match_rate']:.1%})")
    print(f"Close Matches (±4 total points): {agreement['close_matches']} ({agreement['close_match_rate']:.1%})")
    
    print(f"\n💰 COST ANALYSIS")
    cost = comparison["cost_analysis"]
    print(f"OpenAI Total Cost: ${cost['total_openai_cost']:.2f}")
    print(f"OpenAI Cost Per Evaluation: ${cost['avg_cost_per_evaluation']:.4f}")
    print(f"Rule-Based Cost: $0.00 (FREE)")
    print(f"💡 Cost Savings with Rule-Based: ${cost['cost_savings_rule_based']:.2f}")
    
    print(f"\n⚡ PERFORMANCE ANALYSIS")
    perf = comparison["performance_analysis"]
    print(f"OpenAI Total Time: {perf['total_openai_time']:.1f}s")
    print(f"OpenAI Avg Time Per Evaluation: {perf['avg_openai_time_per_evaluation']:.2f}s")
    print(f"Rule-Based Estimated Time: {perf['estimated_rule_time']:.2f}s")
    print(f"🚀 Speed Advantage (Rule-Based): {perf['speed_advantage_rule_based']}")
    
    print(f"\n🎯 RECOMMENDATIONS")
    
    # Calculate overall preference
    avg_diff = statistics.mean([
        abs(diffs["relevancy"]["mean_diff"]),
        abs(diffs["adequacy"]["mean_diff"]),
        abs(diffs["clarity"]["mean_diff"]),
        abs(diffs["consistency"]["mean_diff"])
    ])
    
    if avg_diff > 1.0:
        print("🔥 SIGNIFICANT DIFFERENCES detected between evaluators")
        print("   Consider using OpenAI for higher accuracy in complex scenarios")
        print("   Consider using Rule-Based for consistent, fast, free evaluation")
    elif avg_diff > 0.5:
        print("📊 MODERATE DIFFERENCES detected between evaluators")
        print("   Both evaluators provide reasonable results")
        print("   Choose based on cost/speed vs. accuracy needs")
    else:
        print("✅ SIMILAR RESULTS from both evaluators")
        print("   Rule-Based evaluation is sufficient for most use cases")
        print("   Use OpenAI only when budget allows and nuanced evaluation needed")
    
    if cost["total_openai_cost"] > 5.0:
        print("💸 HIGH COST detected for OpenAI evaluation")
        print("   Consider rule-based evaluation for regular testing")
        print("   Use OpenAI evaluation for final/important assessments only")
    
    print("\n" + "="*80)


def main():
    """Main function to compare both judges."""
    parser = argparse.ArgumentParser(description='Compare rule-based and OpenAI evaluation results')
    parser.add_argument(
        '--results', '-r',
        required=True,
        help='Path to JSONL file containing test results'
    )
    parser.add_argument(
        '--paper', '-p',
        required=True,
        help='Path to text file containing research paper content'
    )
    parser.add_argument(
        '--openai-model',
        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
        default='gpt-3.5-turbo',
        help='OpenAI model to use (default: gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--max-cost',
        type=float,
        default=5.0,
        help='Maximum allowed OpenAI cost in USD (default: 5.0)'
    )
    parser.add_argument(
        '--output-dir',
        default='../data',
        help='Output directory for evaluation files (default: ../data)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load test results and paper content
        logger.info("Loading test results and paper content...")
        test_results = load_test_results_rule(args.results)
        paper_content = load_paper_content_rule(args.paper)
        
        # Estimate OpenAI cost
        estimated_cost = len(test_results) * (0.05 if args.openai_model == 'gpt-4' else 0.02)
        if estimated_cost > args.max_cost:
            logger.error(f"Estimated OpenAI cost ${estimated_cost:.2f} exceeds maximum ${args.max_cost:.2f}")
            return 1
        
        logger.info(f"💰 Estimated OpenAI cost: ${estimated_cost:.2f}")
        
        # Run Rule-Based Evaluation
        logger.info("🤖 Running rule-based evaluation...")
        rule_start_time = time.time()
        rule_evaluator = ResponseEvaluator(paper_content)
        rule_results = rule_evaluator.evaluate_all_responses(test_results)
        rule_time = time.time() - rule_start_time
        
        # Run OpenAI Evaluation
        logger.info("🧠 Running OpenAI evaluation...")
        openai_start_time = time.time()
        openai_evaluator = OpenAIResponseEvaluator(paper_content, model=args.openai_model)
        openai_results = openai_evaluator.evaluate_all_responses(test_results)
        openai_time = time.time() - openai_start_time
        
        logger.info(f"⏱️  Rule-based evaluation: {rule_time:.1f}s")
        logger.info(f"⏱️  OpenAI evaluation: {openai_time:.1f}s")
        
        # Save individual results
        rule_output = os.path.join(args.output_dir, "rule_based_evaluated.jsonl")
        openai_output = os.path.join(args.output_dir, "openai_evaluated.jsonl")
        
        # Save rule-based results
        os.makedirs(os.path.dirname(rule_output), exist_ok=True)
        with open(rule_output, 'w', encoding='utf-8') as f:
            for result in rule_results:
                result_dict = asdict(result)
                result_dict['evaluator'] = 'rule_based'
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        # Save OpenAI results  
        with open(openai_output, 'w', encoding='utf-8') as f:
            for result in openai_results:
                result_dict = asdict(result)
                result_dict['evaluator'] = 'openai'
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        # Compare results
        logger.info("📊 Analyzing comparison...")
        comparison = compare_evaluations(rule_results, openai_results)
        
        # Save comparison report
        comparison_output = os.path.join(args.output_dir, "judge_comparison.json")
        with open(comparison_output, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Print report
        print_comparison_report(comparison)
        
        logger.info(f"📁 Results saved:")
        logger.info(f"   Rule-based: {rule_output}")
        logger.info(f"   OpenAI: {openai_output}")
        logger.info(f"   Comparison: {comparison_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())
