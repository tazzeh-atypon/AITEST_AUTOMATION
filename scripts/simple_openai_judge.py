#!/usr/bin/env python3
"""
Simple OpenAI Judge for Demo

A minimal OpenAI-powered evaluation system for chatbot responses.
No complex features - just basic evaluation using OpenAI.

Author: AI Testing System
Date: 2025-09-27
"""

import json
import os
import openai

# Load environment variables from .env file
from env_loader import load_env

def evaluate_with_openai(question, response, context, api_key=None):
    """
    Simple OpenAI evaluation function.
    
    Args:
        question (str): The question that was asked
        response (str): The chatbot's response
        context (str): The research paper context
        api_key (str): OpenAI API key (optional, uses env var)
    
    Returns:
        dict: Simple evaluation scores and comment
    """
    
    # Set up OpenAI
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai.api_key:
        return {
            "relevancy": 0, "adequacy": 0, "clarity": 0, "consistency": 0,
            "comment": "No OpenAI API key provided"
        }
    
    # Create simple evaluation prompt
    prompt = f"""Evaluate this chatbot response on a scale of 0-5 for each metric:

CONTEXT: {context[:1000]}...

QUESTION: {question}
RESPONSE: {response}

Rate 0-5 for:
- Relevancy: How well does it use the context?
- Adequacy: Is the answer complete?
- Clarity: Is it clear and understandable?
- Consistency: Is it logically consistent?

Return only JSON like this:
{{"relevancy": 4, "adequacy": 3, "clarity": 5, "consistency": 4, "comment": "Brief explanation"}}"""

    try:
        # Simple OpenAI API call
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Ensure all scores are 0-5
        for key in ['relevancy', 'adequacy', 'clarity', 'consistency']:
            if key in result:
                result[key] = max(0, min(5, int(result[key])))
        
        return result
        
    except Exception as e:
        # Simple error handling
        return {
            "relevancy": 0, "adequacy": 0, "clarity": 0, "consistency": 0,
            "comment": f"Evaluation error: {str(e)}"
        }


def evaluate_jsonl_file(input_file, output_file, paper_file):
    """
    Evaluate all responses in a JSONL file using OpenAI.
    
    Args:
        input_file (str): Path to test results JSONL
        output_file (str): Path to save evaluations
        paper_file (str): Path to research paper text
    """
    
    # Load paper content
    with open(paper_file, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    print(f"🧠 Evaluating responses with OpenAI...")
    
    # Process each test result
    with open(input_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                
                try:
                    # Load test result
                    test_data = json.loads(line)
                    
                    question = test_data.get('user_input') or test_data.get('question', '')
                    response = test_data.get('response') or test_data.get('assistant_answer', '')
                    test_id = test_data.get('id', f'test_{line_num}')
                    
                    print(f"   Evaluating {test_id}...")
                    
                    # Get evaluation from OpenAI
                    evaluation = evaluate_with_openai(question, response, paper_content)
                    
                    # Add metadata
                    evaluation['id'] = test_id
                    evaluation['q_type'] = test_data.get('q_type', 'unknown')
                    
                    # Save evaluation
                    fout.write(json.dumps(evaluation, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"   Error on line {line_num}: {str(e)}")
                    continue
    
    print(f"✅ OpenAI evaluation complete! Saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple OpenAI evaluation')
    parser.add_argument('--results', '-r', required=True, help='Test results JSONL file')
    parser.add_argument('--paper', '-p', required=True, help='Research paper text file')
    parser.add_argument('--output', '-o', default='openai_eval.jsonl', help='Output file')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run evaluation
    evaluate_jsonl_file(args.results, args.output, args.paper)
