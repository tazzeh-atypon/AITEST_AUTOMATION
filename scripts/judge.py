#!/usr/bin/env python3
"""
Response Evaluation Script for AI Chatbot Testing

This script evaluates chatbot responses on four key metrics:
- Relevancy: Is the answer based on the research paper?
- Adequacy: Is the answer complete enough?
- Clarity: Is the answer clear and understandable?
- Confusion: Is the answer internally consistent?

Author: AI Testing System
Date: 2025-09-27
"""

import json
import argparse
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class for storing evaluation results."""
    id: str
    relevancy: int
    adequacy: int
    clarity: int
    confusion: int
    comment: str
    response_type: str  # 'answer', 'cannot_answer', 'error', 'invalid'


class ResponseEvaluator:
    """
    Evaluates chatbot responses based on multiple quality metrics.
    
    Provides automated scoring for relevancy, adequacy, clarity, and confusion
    with contextual understanding of question types and expected behaviors.
    """
    
    def __init__(self, paper_content: str):
        """
        Initialize the evaluator with research paper content.
        
        Args:
            paper_content (str): Full text of the research paper for context
        """
        self.paper_content = paper_content.lower()
        self.paper_keywords = self._extract_keywords()
        
    def _extract_keywords(self) -> List[str]:
        """Extract key terms and concepts from the paper for relevancy checking."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = re.findall(r'\b[a-zA-Z]{4,}\b', self.paper_content)
        # Get most common words (excluding common stopwords)
        stopwords = {'that', 'with', 'have', 'this', 'will', 'from', 'they', 
                    'been', 'were', 'said', 'each', 'which', 'their', 'time', 
                    'would', 'there', 'could', 'other', 'more', 'what', 'into',
                    'many', 'some', 'then', 'them', 'these', 'also', 'such'}
        
        word_freq = {}
        for word in words:
            if word.lower() not in stopwords and len(word) > 4:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        
        # Return top 50 keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:50]]
    
    def _classify_response_type(self, response: Optional[str]) -> str:
        """
        Classify the type of response for appropriate evaluation.
        
        Args:
            response (Optional[str]): The chatbot response to classify
            
        Returns:
            str: Response type classification
        """
        if not response:
            return 'error'
        
        response_lower = response.lower()
        
        # Check for "cannot answer" type responses
        cannot_answer_patterns = [
            r'cannot answer',
            r'unable to answer',
            r'don\'t have.*information',
            r'not.*provided.*paper',
            r'cannot find.*information',
            r'not.*mentioned.*paper',
            r'sorry.*cannot',
            r'insufficient.*information',
            r'not.*available.*text',
            r'cannot determine'
        ]
        
        for pattern in cannot_answer_patterns:
            if re.search(pattern, response_lower):
                return 'cannot_answer'
        
        # Check if response seems like an actual answer attempt
        if len(response.strip()) < 10:
            return 'invalid'
        
        return 'answer'
    
    def _evaluate_relevancy(self, response: str, question_type: str, 
                           expected_behavior: str) -> Tuple[int, str]:
        """
        Evaluate how well the response relates to the research paper content.
        
        Args:
            response (str): Chatbot response
            question_type (str): Type of question asked
            expected_behavior (str): Expected behavior description
            
        Returns:
            Tuple[int, str]: (score 0-5, explanation)
        """
        response_lower = response.lower()
        
        # For "not_answerable" questions, appropriate rejection is highly relevant
        if question_type == 'not_answerable':
            response_type = self._classify_response_type(response)
            if response_type == 'cannot_answer':
                return 5, "Correctly identified that question cannot be answered from paper"
            elif response_type == 'answer':
                return 1, "Incorrectly attempted to answer question not covered in paper"
            else:
                return 0, "Invalid or error response to unanswerable question"
        
        # Count paper-related keywords in response
        keyword_matches = sum(1 for keyword in self.paper_keywords 
                            if keyword in response_lower)
        keyword_ratio = keyword_matches / max(len(self.paper_keywords), 1)
        
        # Check for generic/template responses
        generic_patterns = [
            r'according to.*paper',
            r'research shows',
            r'study indicates',
            r'findings suggest'
        ]
        has_paper_references = any(re.search(pattern, response_lower) 
                                 for pattern in generic_patterns)
        
        # Scoring logic
        if keyword_ratio > 0.1 and has_paper_references:
            score = 5
            explanation = "Response contains relevant paper content with appropriate references"
        elif keyword_ratio > 0.05:
            score = 4
            explanation = "Response contains some relevant paper content"
        elif has_paper_references:
            score = 3
            explanation = "Response references paper but limited content overlap"
        elif len(response) > 50 and not re.search(r'cannot.*answer', response_lower):
            score = 2
            explanation = "Response attempts to answer but shows little paper relevance"
        else:
            score = 1
            explanation = "Response shows minimal relevance to paper content"
        
        return score, explanation
    
    def _evaluate_adequacy(self, response: str, question_type: str, 
                          user_input: str) -> Tuple[int, str]:
        """
        Evaluate how complete and sufficient the response is.
        
        Args:
            response (str): Chatbot response
            question_type (str): Type of question asked
            user_input (str): Original user question
            
        Returns:
            Tuple[int, str]: (score 0-5, explanation)
        """
        response_type = self._classify_response_type(response)
        
        # For cannot_answer responses, adequacy depends on appropriateness
        if response_type == 'cannot_answer':
            if question_type == 'not_answerable':
                return 5, "Adequate response - correctly declined to answer"
            else:
                return 2, "Inadequate - declined to answer answerable question"
        
        # For error responses
        if response_type == 'error' or response_type == 'invalid':
            return 0, "Inadequate - no meaningful response provided"
        
        # Evaluate answer completeness
        response_length = len(response.strip())
        
        # Formatting questions expect specific formats
        if question_type == 'formatting':
            format_indicators = ['summary:', 'list:', 'steps:', '1.', '•', '-']
            has_formatting = any(indicator in response.lower() 
                               for indicator in format_indicators)
            
            if has_formatting and response_length > 100:
                score = 5
                explanation = "Complete response with appropriate formatting"
            elif has_formatting:
                score = 4
                explanation = "Formatted response but could be more detailed"
            elif response_length > 100:
                score = 3
                explanation = "Detailed response but lacks requested formatting"
            else:
                score = 2
                explanation = "Response lacks both detail and formatting"
        
        # Regular questions
        else:
            if response_length > 200:
                score = 5
                explanation = "Comprehensive and detailed response"
            elif response_length > 100:
                score = 4
                explanation = "Good level of detail in response"
            elif response_length > 50:
                score = 3
                explanation = "Basic response with some detail"
            elif response_length > 20:
                score = 2
                explanation = "Very brief response, lacks detail"
            else:
                score = 1
                explanation = "Minimal response, inadequate detail"
        
        return score, explanation
    
    def _evaluate_clarity(self, response: str) -> Tuple[int, str]:
        """
        Evaluate how clear and understandable the response is.
        
        Args:
            response (str): Chatbot response
            
        Returns:
            Tuple[int, str]: (score 0-5, explanation)
        """
        if not response or len(response.strip()) < 5:
            return 0, "No meaningful response to evaluate"
        
        response_type = self._classify_response_type(response)
        
        if response_type == 'cannot_answer':
            return 5, "Clear and direct communication of inability to answer"
        
        # Check for clarity indicators
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Check for unclear patterns
        unclear_patterns = [
            r'um+\s',
            r'uh+\s',
            r'well+\s*,?\s*',
            r'like\s*,?\s*',
            r'you know',
            r'sort of',
            r'kind of'
        ]
        
        has_unclear_language = any(re.search(pattern, response.lower()) 
                                 for pattern in unclear_patterns)
        
        # Check for structure indicators
        structure_indicators = [
            r'first\w*\s*,?',
            r'second\w*\s*,?',
            r'finally\s*,?',
            r'in conclusion',
            r'therefore',
            r'however',
            r'moreover'
        ]
        
        has_structure = any(re.search(pattern, response.lower()) 
                          for pattern in structure_indicators)
        
        # Scoring logic
        if has_unclear_language:
            score = 2
            explanation = "Response contains unclear language or filler words"
        elif avg_sentence_length > 30:
            score = 3
            explanation = "Response has very long sentences that may be hard to follow"
        elif has_structure and avg_sentence_length < 25:
            score = 5
            explanation = "Clear, well-structured response with appropriate sentence length"
        elif avg_sentence_length < 25:
            score = 4
            explanation = "Clear response with good sentence structure"
        else:
            score = 3
            explanation = "Generally clear but could be better structured"
        
        return score, explanation
    
    def _evaluate_confusion(self, response: str) -> Tuple[int, str]:
        """
        Evaluate internal consistency and logical coherence of the response.
        
        Args:
            response (str): Chatbot response
            
        Returns:
            Tuple[int, str]: (score 0-5, explanation)
        """
        if not response or len(response.strip()) < 5:
            return 0, "No meaningful response to evaluate"
        
        response_type = self._classify_response_type(response)
        
        if response_type == 'cannot_answer':
            return 5, "Consistent message about inability to answer"
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'however\b', r'therefore\b'),
            (r'but\b', r'thus\b'),
            (r'although\b', r'because\b'),
            (r'not\s+\w+', r'is\s+\w+'),
        ]
        
        has_contradictions = False
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, response.lower()) and re.search(pattern2, response.lower()):
                # This is a simple heuristic - more sophisticated analysis could be added
                sentences = response.split('.')
                if len(sentences) > 1:
                    # Check if contradictory terms appear in adjacent sentences
                    for i in range(len(sentences) - 1):
                        if (re.search(pattern1, sentences[i].lower()) and 
                            re.search(pattern2, sentences[i+1].lower())):
                            has_contradictions = True
                            break
        
        # Check for repeated information
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if len(sentences) > 2:
            # Simple check for highly similar sentences (potential repetition)
            for i, sent1 in enumerate(sentences):
                for sent2 in sentences[i+1:]:
                    if len(sent1) > 20 and len(sent2) > 20:
                        # Calculate simple similarity (shared words)
                        words1 = set(sent1.lower().split())
                        words2 = set(sent2.lower().split())
                        if words1 and words2:
                            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                            if similarity > 0.7:
                                has_contradictions = True
                                break
        
        # Check for logical flow
        transition_words = ['therefore', 'thus', 'consequently', 'however', 'moreover', 
                          'furthermore', 'additionally', 'in contrast', 'similarly']
        has_transitions = any(word in response.lower() for word in transition_words)
        
        # Scoring logic
        if has_contradictions:
            score = 1
            explanation = "Response contains contradictory or repetitive information"
        elif not has_transitions and len(sentences) > 3:
            score = 3
            explanation = "Response lacks logical transitions between ideas"
        elif has_transitions:
            score = 5
            explanation = "Response shows good logical flow and consistency"
        else:
            score = 4
            explanation = "Response is generally consistent and coherent"
        
        return score, explanation
    
    def evaluate_response(self, test_result: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single test result across all metrics.
        
        Args:
            test_result (Dict): Test result containing response and metadata
            
        Returns:
            EvaluationResult: Complete evaluation with scores and comments
        """
        response = test_result.get('response', '')
        question_type = test_result.get('q_type', 'unknown')
        user_input = test_result.get('user_input', '')
        expected_behavior = test_result.get('expected_behavior', '')
        test_id = test_result.get('id', 'unknown')
        
        logger.info(f"Evaluating response for test {test_id}")
        
        # Classify response type
        response_type = self._classify_response_type(response)
        
        # Evaluate each metric
        relevancy_score, relevancy_comment = self._evaluate_relevancy(
            response, question_type, expected_behavior
        )
        
        adequacy_score, adequacy_comment = self._evaluate_adequacy(
            response, question_type, user_input
        )
        
        clarity_score, clarity_comment = self._evaluate_clarity(response)
        
        confusion_score, confusion_comment = self._evaluate_confusion(response)
        
        # Combine comments
        overall_comment = f"Relevancy: {relevancy_comment}. Adequacy: {adequacy_comment}. Clarity: {clarity_comment}. Consistency: {confusion_comment}."
        
        return EvaluationResult(
            id=test_id,
            relevancy=relevancy_score,
            adequacy=adequacy_score,
            clarity=clarity_score,
            confusion=confusion_score,
            comment=overall_comment,
            response_type=response_type
        )
    
    def evaluate_all_responses(self, test_results: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """
        Evaluate all test results.
        
        Args:
            test_results (List[Dict]): List of test results to evaluate
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        logger.info(f"Evaluating {len(test_results)} responses")
        
        evaluations = []
        for test_result in test_results:
            try:
                evaluation = self.evaluate_response(test_result)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating test {test_result.get('id', 'unknown')}: {str(e)}")
                
                # Create error evaluation
                error_evaluation = EvaluationResult(
                    id=test_result.get('id', 'unknown'),
                    relevancy=0,
                    adequacy=0,
                    clarity=0,
                    confusion=0,
                    comment=f"Evaluation error: {str(e)}",
                    response_type='error'
                )
                evaluations.append(error_evaluation)
        
        logger.info(f"Completed evaluation of {len(evaluations)} responses")
        return evaluations


def load_test_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test results from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of test result dictionaries
    """
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
    """
    Load research paper content from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Paper content
    """
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


def save_evaluations(evaluations: List[EvaluationResult], output_file: str) -> None:
    """
    Save evaluation results to a JSONL file.
    
    Args:
        evaluations (List[EvaluationResult]): Evaluation results to save
        output_file (str): Path to output file
    """
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
                    'response_type': evaluation.response_type
                }
                
                json_line = json.dumps(eval_dict, ensure_ascii=False)
                file.write(json_line + '\n')
        
        logger.info(f"Evaluations saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving evaluations to {output_file}: {str(e)}")
        raise


def main():
    """Main function to evaluate responses from command line."""
    parser = argparse.ArgumentParser(description='Evaluate AI chatbot responses')
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
        default='../data/evaluated.jsonl',
        help='Path to output JSONL file (default: ../data/evaluated.jsonl)'
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
        test_results = load_test_results(args.results)
        paper_content = load_paper_content(args.paper)
        
        # Initialize evaluator
        evaluator = ResponseEvaluator(paper_content)
        
        # Evaluate responses
        evaluations = evaluator.evaluate_all_responses(test_results)
        
        # Save evaluations
        save_evaluations(evaluations, args.output)
        
        # Print summary
        avg_relevancy = sum(e.relevancy for e in evaluations) / len(evaluations)
        avg_adequacy = sum(e.adequacy for e in evaluations) / len(evaluations)
        avg_clarity = sum(e.clarity for e in evaluations) / len(evaluations)
        avg_confusion = sum(e.confusion for e in evaluations) / len(evaluations)
        
        print(f"✅ Evaluation completed!")
        print(f"📊 Average Scores:")
        print(f"   Relevancy: {avg_relevancy:.2f}/5")
        print(f"   Adequacy:  {avg_adequacy:.2f}/5")
        print(f"   Clarity:   {avg_clarity:.2f}/5")
        print(f"   Confusion: {avg_confusion:.2f}/5")
        print(f"📁 Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
