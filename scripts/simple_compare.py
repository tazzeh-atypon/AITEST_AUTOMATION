#!/usr/bin/env python3
"""
Simple Judge Comparison for Demo

Compares rule-based vs OpenAI evaluation in a simple, easy-to-understand way.

Author: AI Testing System
Date: 2025-09-27
"""

import json
import os

def compare_judges(rule_file, openai_file):
    """
    Simple comparison between rule-based and OpenAI evaluations.
    
    Args:
        rule_file (str): Path to rule-based evaluation JSONL
        openai_file (str): Path to OpenAI evaluation JSONL
    """
    
    print("🔬 SIMPLE JUDGE COMPARISON")
    print("=" * 50)
    
    # Load results
    rule_results = []
    openai_results = []
    
    with open(rule_file, 'r') as f:
        for line in f:
            if line.strip():
                rule_results.append(json.loads(line))
    
    with open(openai_file, 'r') as f:
        for line in f:
            if line.strip():
                openai_results.append(json.loads(line))
    
    print(f"📊 Loaded {len(rule_results)} rule-based and {len(openai_results)} OpenAI evaluations")
    
    # Calculate averages
    rule_avg = {
        'relevancy': sum(r['relevancy'] for r in rule_results) / len(rule_results),
        'adequacy': sum(r['adequacy'] for r in rule_results) / len(rule_results),
        'clarity': sum(r['clarity'] for r in rule_results) / len(rule_results),
        'consistency': sum(r.get('confusion', r.get('consistency', 0)) for r in rule_results) / len(rule_results)
    }
    
    openai_avg = {
        'relevancy': sum(r['relevancy'] for r in openai_results) / len(openai_results),
        'adequacy': sum(r['adequacy'] for r in openai_results) / len(openai_results),
        'clarity': sum(r['clarity'] for r in openai_results) / len(openai_results),
        'consistency': sum(r.get('consistency', r.get('confusion', 0)) for r in openai_results) / len(openai_results)
    }
    
    print(f"\n🤖 RULE-BASED AVERAGES:")
    for metric, score in rule_avg.items():
        print(f"   {metric.capitalize()}: {score:.2f}/5")
    
    print(f"\n🧠 OPENAI AVERAGES:")
    for metric, score in openai_avg.items():
        print(f"   {metric.capitalize()}: {score:.2f}/5")
    
    print(f"\n📈 DIFFERENCES (OpenAI - Rule-Based):")
    total_diff = 0
    for metric in rule_avg:
        diff = openai_avg[metric] - rule_avg[metric]
        total_diff += abs(diff)
        symbol = "📈" if diff > 0.5 else "📉" if diff < -0.5 else "➡️"
        print(f"   {symbol} {metric.capitalize()}: {diff:+.2f}")
    
    avg_diff = total_diff / len(rule_avg)
    
    print(f"\n🎯 SUMMARY:")
    if avg_diff > 1.0:
        print("   🔥 MAJOR DIFFERENCES - OpenAI evaluation is significantly different")
        print("   💡 Use OpenAI for more nuanced evaluation")
    elif avg_diff > 0.3:
        print("   📊 MODERATE DIFFERENCES - Both evaluators give different perspectives")
        print("   💡 Choose based on your needs: Speed vs Accuracy")
    else:
        print("   ✅ SIMILAR RESULTS - Both evaluators largely agree")
        print("   💡 Rule-based evaluation is sufficient for most cases")
    
    print(f"\n💰 COST CONSIDERATION:")
    estimated_cost = len(openai_results) * 0.02  # Rough estimate
    print(f"   OpenAI cost: ~${estimated_cost:.2f}")
    print(f"   Rule-based cost: $0.00 (FREE)")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare rule-based vs OpenAI evaluation')
    parser.add_argument('--rule', required=True, help='Rule-based evaluation JSONL file')
    parser.add_argument('--openai', required=True, help='OpenAI evaluation JSONL file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.rule):
        print(f"❌ Rule-based file not found: {args.rule}")
        exit(1)
    
    if not os.path.exists(args.openai):
        print(f"❌ OpenAI file not found: {args.openai}")
        exit(1)
    
    compare_judges(args.rule, args.openai)
