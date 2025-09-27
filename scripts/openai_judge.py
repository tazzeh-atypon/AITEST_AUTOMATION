#!/usr/bin/env python3
"""
OpenAI-Powered Response Evaluation Script for AI Chatbot Testing

This script uses OpenAI's GPT models to evaluate chatbot responses with
human-like understanding and nuanced assessment capabilities.

Author: AI Testing System  
Date: 2025-09-27
"""

import json
import argparse
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OpenAIEvaluationResult:
    """Data class for storing OpenAI evaluation results."""
    id: str
    relevancy: int
    adequacy: int
    clarity: int
    confusion: int
    comment: str
    response_type: str
    evaluation_cost: float
    evaluation_time: float


class OpenAIResponseEvaluator:
    """
    Evaluates chatbot responses using OpenAI's GPT models for more nuanced,
    human-like assessment of response quality.
    """
    
    def __init__(self, paper_content: str, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the OpenAI evaluator.
        
        Args:
            paper_content (str): Full text of the research paper
            model (str): OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
            max_retries (int): Maximum retry attempts for API calls
        """
        self.paper_content = paper_content
        self.model = model
        self.max_retries = max_retries
        self.total_cost = 0.0
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Cost tracking (approximate costs per 1K tokens)
        self.cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
    
    def _create_evaluation_prompt(self, question: str, response: str, 
                                question_type: str, expected_behavior: str) -> str:
        """Create a comprehensive evaluation prompt for OpenAI."""
        
        prompt = f"""You are an expert AI system evaluator tasked with assessing chatbot responses based on a research paper. 

RESEARCH PAPER CONTENT:
{self.paper_content[:2000]}...

QUESTION TYPE: {question_type}
USER QUESTION: {question}
CHATBOT RESPONSE: {response}
EXPECTED BEHAVIOR: {expected_behavior}

Please evaluate the chatbot response on these four metrics (0-5 scale):

1. RELEVANCY (0-5): How well does the response relate to and use information from the research paper?
   - 5: Directly uses relevant paper content with accurate references
   - 4: Uses mostly relevant paper content with minor gaps
   - 3: Some relevant paper content but with significant gaps
   - 2: Limited use of paper content, mostly generic
   - 1: Minimal connection to paper content
   - 0: No connection to paper or completely wrong

2. ADEQUACY (0-5): Is the response complete and sufficient for the question asked?
   - 5: Comprehensive, fully answers the question
   - 4: Good coverage, minor gaps in completeness
   - 3: Adequate but could be more complete
   - 2: Partial answer, missing important elements
   - 1: Very incomplete, major gaps
   - 0: No meaningful answer provided

3. CLARITY (0-5): How clear and understandable is the response?
   - 5: Exceptionally clear, well-structured, easy to understand
   - 4: Clear and well-organized
   - 3: Generally clear with minor issues
   - 2: Somewhat unclear or poorly organized
   - 1: Difficult to understand
   - 0: Confusing or incomprehensible

4. CONSISTENCY (0-5): Is the response internally consistent and logically coherent?
   - 5: Perfectly consistent, excellent logical flow
   - 4: Consistent with good logical flow
   - 3: Generally consistent with minor issues
   - 2: Some inconsistencies or logical gaps
   - 1: Notable inconsistencies
   - 0: Major contradictions or logical failures

SPECIAL CONSIDERATIONS:
- For "not_answerable" questions: High scores if properly declines to answer
- For "formatting" questions: Consider if proper format is used
- For "history" questions: Evaluate context awareness

Please respond with ONLY a JSON object in this exact format:
{{
    "relevancy": <score>,
    "adequacy": <score>, 
    "clarity": <score>,
    "consistency": <score>,
    "comment": "<brief explanation of scores and key observations>",
    "response_type": "<answer|cannot_answer|error|invalid>"
}}"""
        
        return prompt
    
    def _estimate_cost(self, prompt: str, response_text: str) -> float:
        """Estimate API call cost based on token usage."""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0
        
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) / 4
        output_tokens = len(response_text) / 4
        
        costs = self.cost_per_1k_tokens[self.model]
        estimated_cost = (
            (input_tokens / 1000) * costs["input"] + 
            (output_tokens / 1000) * costs["output"]
        )
        
        return estimated_cost
    
    def _call_openai_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Make OpenAI API call with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert AI evaluation system."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=500
                )
                
                evaluation_time = time.time() - start_time
                response_text = response.choices[0].message.content
                
                # Estimate cost
                estimated_cost = self._estimate_cost(prompt, response_text)
                self.total_cost += estimated_cost
                
                # Parse JSON response
                evaluation_data = json.loads(response_text)
                evaluation_data["_meta"] = {
                    "cost": estimated_cost,
                    "time": evaluation_time,
                    "tokens_estimated": len(prompt + response_text) / 4
                }
                
                return evaluation_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: JSON decode error - {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "relevancy": 0, "adequacy": 0, "clarity": 0, "consistency": 0,
                        "comment": f"JSON parsing failed: {str(e)}",
                        "response_type": "error",
                        "_meta": {"cost": 0.0, "time": 0.0, "tokens_estimated": 0}
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except openai.RateLimitError as e:
                logger.warning(f"Attempt {attempt + 1}: Rate limit error - waiting...")
                time.sleep(min(60, 2 ** attempt))
                
            except openai.APIError as e:
                logger.error(f"Attempt {attempt + 1}: OpenAI API error - {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "relevancy": 0, "adequacy": 0, "clarity": 0, "consistency": 0,
                        "comment": f"API error: {str(e)}",
                        "response_type": "error", 
                        "_meta": {"cost": 0.0, "time": 0.0, "tokens_estimated": 0}
                    }
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Unexpected error - {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "relevancy": 0, "adequacy": 0, "clarity": 0, "consistency": 0,
                        "comment": f"Unexpected error: {str(e)}",
                        "response_type": "error",
                        "_meta": {"cost": 0.0, "time": 0.0, "tokens_estimated": 0}
                    }
                time.sleep(2 ** attempt)
    
    def evaluate_response(self, test_result: Dict[str, Any]) -> OpenAIEvaluationResult:
        """
        Evaluate a single test result using OpenAI.
        
        Args:
            test_result (Dict): Test result containing response and metadata
            
        Returns:
            OpenAIEvaluationResult: Complete evaluation with scores and metadata
        """
        response = test_result.get('response', '')
        question_type = test_result.get('q_type', 'unknown')
        user_input = test_result.get('user_input', '')
        expected_behavior = test_result.get('expected_behavior', '')
        test_id = test_result.get('id', 'unknown')
        
        logger.info(f"Evaluating response for test {test_id} using OpenAI ({self.model})")
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            user_input, response, question_type, expected_behavior
        )
        
        # Get evaluation from OpenAI
        evaluation_data = self._call_openai_with_retry(prompt)
        meta = evaluation_data.pop("_meta", {})
        
        # Validate scores are in range 0-5
        for metric in ['relevancy', 'adequacy', 'clarity', 'consistency']:
            if metric in evaluation_data:
                evaluation_data[metric] = max(0, min(5, int(evaluation_data[metric])))
        
        return OpenAIEvaluationResult(
            id=test_id,
            relevancy=evaluation_data.get('relevancy', 0),
            adequacy=evaluation_data.get('adequacy', 0), 
            clarity=evaluation_data.get('clarity', 0),
            confusion=evaluation_data.get('consistency', 0),  # Note: using 'consistency' for 'confusion'
            comment=evaluation_data.get('comment', 'No comment provided'),
            response_type=evaluation_data.get('response_type', 'unknown'),
            evaluation_cost=meta.get('cost', 0.0),
            evaluation_time=meta.get('time', 0.0)
        )
    
    def evaluate_all_responses(self, test_results: List[Dict[str, Any]]) -> List[OpenAIEvaluationResult]:
        """
        Evaluate all test results using OpenAI.
        
        Args:
            test_results (List[Dict]): List of test results to evaluate
            
        Returns:
            List[OpenAIEvaluationResult]: List of evaluation results
        """
        logger.info(f"Evaluating {len(test_results)} responses using OpenAI {self.model}")
        logger.info(f"Estimated cost: ${len(test_results) * 0.02:.2f} - ${len(test_results) * 0.08:.2f}")
        
        evaluations = []
        total_time = 0.0
        
        for i, test_result in enumerate(test_results, 1):
            try:
                evaluation = self.evaluate_response(test_result)
                evaluations.append(evaluation)
                total_time += evaluation.evaluation_time
                
                logger.info(f"✅ Evaluated {i}/{len(test_results)} - Cost: ${evaluation.evaluation_cost:.4f}")
                
                # Rate limiting - add small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error evaluating test {test_result.get('id', 'unknown')}: {str(e)}")
                
                # Create error evaluation
                error_evaluation = OpenAIEvaluationResult(
                    id=test_result.get('id', 'unknown'),
                    relevancy=0, adequacy=0, clarity=0, confusion=0,
                    comment=f"Evaluation error: {str(e)}",
                    response_type='error',
                    evaluation_cost=0.0, evaluation_time=0.0
                )
                evaluations.append(error_evaluation)
        
        logger.info(f"✅ OpenAI evaluation completed!")
        logger.info(f"💰 Total estimated cost: ${self.total_cost:.2f}")
        logger.info(f"⏱️  Total evaluation time: {total_time:.1f}s")
        
        return evaluations


def load_test_results(file_path: str) -> List[Dict[str, Any]]:
    """Load test results from JSONL file."""
    test_results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    result = json.loads(line)
                    test_results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(test_results)} test results from {file_path}")
        return test_results
        
    except FileNotFoundError:
        logger.error(f"Test results file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading test results from {file_path}: {str(e)}")
        raise


def load_paper_content(file_path: str) -> str:
    """Load research paper content from text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        if not content:
            raise ValueError(f"File {file_path} is empty")
        
        logger.info(f"Loaded paper content: {len(content)} characters")
        return content
        
    except FileNotFoundError:
        logger.error(f"Paper content file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading paper content from {file_path}: {str(e)}")
        raise


def save_evaluations(evaluations: List[OpenAIEvaluationResult], output_file: str) -> None:
    """Save OpenAI evaluation results to JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            for evaluation in evaluations:
                eval_dict = {
                    'id': evaluation.id,
                    'relevancy': evaluation.relevancy,
                    'adequacy': evaluation.adequacy,
                    'clarity': evaluation.clarity,
                    'confusion': evaluation.confusion,
                    'comment': evaluation.comment,
                    'response_type': evaluation.response_type,
                    'evaluation_cost': evaluation.evaluation_cost,
                    'evaluation_time': evaluation.evaluation_time,
                    'evaluator': 'openai'
                }
                
                json_line = json.dumps(eval_dict, ensure_ascii=False)
                file.write(json_line + '\n')
        
        logger.info(f"OpenAI evaluations saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving evaluations to {output_file}: {str(e)}")
        raise


def main():
    """Main function to evaluate responses using OpenAI."""
    parser = argparse.ArgumentParser(description='Evaluate AI chatbot responses using OpenAI')
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
        '--output', '-o',
        default='../data/openai_evaluated.jsonl',
        help='Path to output JSONL file (default: ../data/openai_evaluated.jsonl)'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
        default='gpt-3.5-turbo',
        help='OpenAI model to use (default: gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--max-cost',
        type=float,
        default=10.0,
        help='Maximum allowed cost in USD (default: 10.0)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return 1
    
    try:
        # Load test results and paper content
        test_results = load_test_results(args.results)
        paper_content = load_paper_content(args.paper)
        
        # Estimate total cost
        estimated_cost = len(test_results) * (0.05 if args.model == 'gpt-4' else 0.02)
        
        if estimated_cost > args.max_cost:
            logger.error(f"Estimated cost ${estimated_cost:.2f} exceeds maximum ${args.max_cost:.2f}")
            logger.error("Use --max-cost to increase limit or reduce number of test results")
            return 1
        
        logger.info(f"💰 Estimated cost: ${estimated_cost:.2f} (max: ${args.max_cost:.2f})")
        
        # Initialize evaluator
        evaluator = OpenAIResponseEvaluator(paper_content, model=args.model)
        
        # Evaluate responses
        evaluations = evaluator.evaluate_all_responses(test_results)
        
        # Save evaluations
        save_evaluations(evaluations, args.output)
        
        # Print summary
        total_cost = sum(e.evaluation_cost for e in evaluations)
        avg_relevancy = sum(e.relevancy for e in evaluations) / len(evaluations)
        avg_adequacy = sum(e.adequacy for e in evaluations) / len(evaluations)
        avg_clarity = sum(e.clarity for e in evaluations) / len(evaluations)
        avg_consistency = sum(e.confusion for e in evaluations) / len(evaluations)
        
        print(f"✅ OpenAI evaluation completed!")
        print(f"💰 Total cost: ${total_cost:.2f}")
        print(f"📊 Average Scores:")
        print(f"   Relevancy: {avg_relevancy:.2f}/5")
        print(f"   Adequacy:  {avg_adequacy:.2f}/5")
        print(f"   Clarity:   {avg_clarity:.2f}/5")
        print(f"   Consistency: {avg_consistency:.2f}/5")
        print(f"📁 Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"OpenAI evaluation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
