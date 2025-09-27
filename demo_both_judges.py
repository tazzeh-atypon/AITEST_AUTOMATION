#!/usr/bin/env python3
"""
Demo: Compare Rule-Based vs OpenAI Evaluation

Simple demo script to test both evaluation approaches and see the differences.

Usage:
    python3 demo_both_judges.py

Requirements:
    - Set OPENAI_API_KEY environment variable
    - Test results file (data/raw_results.jsonl)
    - Research paper file (data/sample_paper.txt)
"""

import os
import json

def run_demo():
    """Run a simple demo comparing both judges."""
    
    print("🎯 AI CHATBOT JUDGE COMPARISON DEMO")
    print("=" * 50)
    
    # Check requirements
    if not os.path.exists("data/raw_results.jsonl"):
        print("❌ Missing data/raw_results.jsonl")
        print("   Run test generation and execution first!")
        return
    
    if not os.path.exists("data/sample_paper.txt"):
        print("❌ Missing data/sample_paper.txt")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found - will skip OpenAI evaluation")
        openai_available = False
    else:
        openai_available = True
    
    print("✅ All requirements met!")
    print()
    
    # Step 1: Rule-based evaluation
    print("🤖 STEP 1: Running Rule-Based Evaluation...")
    os.system("python3 scripts/judge.py -r data/raw_results.jsonl -p data/sample_paper.txt -o data/rule_eval.jsonl")
    print()
    
    # Step 2: OpenAI evaluation (if available)
    if openai_available:
        print("🧠 STEP 2: Running OpenAI Evaluation...")
        os.system("python3 scripts/simple_openai_judge.py -r data/raw_results.jsonl -p data/sample_paper.txt -o data/openai_eval.jsonl")
        print()
        
        # Step 3: Comparison
        print("🔬 STEP 3: Comparing Results...")
        os.system("python3 scripts/simple_compare.py --rule data/rule_eval.jsonl --openai data/openai_eval.jsonl")
    else:
        print("⏭️  STEP 2: Skipped OpenAI evaluation (no API key)")
        print()
        
        # Just show rule-based results
        print("📊 RULE-BASED RESULTS:")
        with open("data/rule_eval.jsonl", 'r') as f:
            results = [json.loads(line) for line in f if line.strip()]
        
        avg_scores = {
            'relevancy': sum(r['relevancy'] for r in results) / len(results),
            'adequacy': sum(r['adequacy'] for r in results) / len(results),
            'clarity': sum(r['clarity'] for r in results) / len(results),
            'consistency': sum(r.get('confusion', 0) for r in results) / len(results)
        }
        
        for metric, score in avg_scores.items():
            print(f"   {metric.capitalize()}: {score:.2f}/5")
    
    print("\n🎉 Demo Complete!")
    print("\n💡 TO USE OPENAI EVALUATION:")
    print("   1. Get OpenAI API key from https://platform.openai.com")
    print("   2. Set environment variable: export OPENAI_API_KEY='your-key'")
    print("   3. Run demo again to see comparison")


if __name__ == "__main__":
    run_demo()
