#!/usr/bin/env python3
"""
Test Runner Script for AI Chatbot Testing

This script executes test cases against a backend API endpoint that uses the Gemini API,
collecting responses and saving them in a structured format for evaluation.

Author: AI Testing System
Date: 2025-09-27
"""

import json
import argparse
import logging
import os
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import urljoin
import concurrent.futures
from dataclasses import dataclass

# Load environment variables from .env file
from env_loader import load_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Data class for storing test results."""
    id: str
    q_type: str
    user_input: str
    expected_behavior: str
    response: Optional[str]
    status: str  # 'success', 'error', 'timeout', 'invalid_response'
    error_message: Optional[str]
    response_time: float
    conversation_history: Optional[List[Dict[str, str]]] = None


class APITestRunner:
    """
    Test runner for executing chatbot tests against a backend API endpoint.
    
    Handles API communication, error recovery, and response collection
    with support for conversation history and various question types.
    """
    
    def __init__(self, 
                 api_endpoint: str,
                 paper_content: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 concurrent_requests: int = 5):
        """
        Initialize the test runner.
        
        Args:
            api_endpoint (str): Base URL of the backend API
            paper_content (str): Full text of the research paper
            api_key (str, optional): API key for authentication
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum number of retry attempts
            concurrent_requests (int): Maximum concurrent API requests
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.paper_content = paper_content
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrent_requests = concurrent_requests
        self.session = requests.Session()
        
        # Set up authentication if API key provided
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.headers.update({
                'Content-Type': 'application/json'
            })
    
    def _prepare_request_payload(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the request payload for the API call.
        
        Args:
            test_case (Dict): Test case containing question and metadata
            
        Returns:
            Dict: Request payload for the API
        """
        payload = {
            'user_input': test_case['user_input'],
            'paper_content': self.paper_content,
            'q_type': test_case.get('q_type', 'unknown'),
            'test_id': test_case.get('id', 'unknown')
        }
        
        # Include conversation history for multi-turn tests
        if 'conversation_history' in test_case and test_case['conversation_history']:
            payload['conversation_history'] = test_case['conversation_history']
        
        return payload
    
    def _make_api_request(self, payload: Dict[str, Any]) -> Tuple[Optional[str], str, float]:
        """
        Make a single API request with retry logic.
        
        Args:
            payload (Dict): Request payload
            
        Returns:
            Tuple[Optional[str], str, float]: (response_text, status, response_time)
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making API request (attempt {attempt + 1})")
                
                response = self.session.post(
                    f"{self.api_endpoint}/chat",
                    json=payload,
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if 'response' in response_data:
                            return response_data['response'], 'success', response_time
                        elif 'message' in response_data:
                            return response_data['message'], 'success', response_time
                        else:
                            return str(response_data), 'success', response_time
                    except json.JSONDecodeError:
                        # If JSON parsing fails, return raw text
                        return response.text, 'success', response_time
                
                elif response.status_code == 429:
                    # Rate limiting - wait and retry
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code >= 500:
                    # Server error - retry
                    last_error = f"Server error: {response.status_code}"
                    time.sleep(min(2 ** attempt, 10))
                    continue
                
                else:
                    # Client error - don't retry
                    return None, f"HTTP {response.status_code}: {response.text}", response_time
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                logger.warning(f"Connection error (attempt {attempt + 1})")
                time.sleep(min(2 ** attempt, 10))
                
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                logger.warning(f"Request error: {str(e)} (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error: {str(e)}")
                break
        
        # All retries exhausted
        response_time = time.time() - start_time
        return None, last_error or "Max retries exhausted", response_time
    
    def _run_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case (Dict): Test case to execute
            
        Returns:
            TestResult: Result of the test execution
        """
        logger.info(f"Running test {test_case.get('id', 'unknown')} ({test_case.get('q_type', 'unknown')})")
        
        # Prepare request payload
        payload = self._prepare_request_payload(test_case)
        
        # Make API request
        response_text, status, response_time = self._make_api_request(payload)
        
        # Create result object
        result = TestResult(
            id=test_case.get('id', 'unknown'),
            q_type=test_case.get('q_type', 'unknown'),
            user_input=test_case.get('user_input', ''),
            expected_behavior=test_case.get('expected_behavior', ''),
            response=response_text,
            status=status,
            error_message=status if status != 'success' else None,
            response_time=response_time,
            conversation_history=test_case.get('conversation_history')
        )
        
        if status == 'success':
            logger.info(f"✅ Test {result.id} completed successfully ({response_time:.2f}s)")
        else:
            logger.error(f"❌ Test {result.id} failed: {status}")
        
        return result
    
    def run_tests(self, test_cases: List[Dict[str, Any]], 
                  use_concurrent: bool = True) -> List[TestResult]:
        """
        Run all test cases.
        
        Args:
            test_cases (List[Dict]): List of test cases to execute
            use_concurrent (bool): Whether to use concurrent execution
            
        Returns:
            List[TestResult]: Results of all test executions
        """
        logger.info(f"Starting execution of {len(test_cases)} test cases")
        start_time = time.time()
        
        results = []
        
        if use_concurrent and self.concurrent_requests > 1:
            # Execute tests concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
                future_to_test = {
                    executor.submit(self._run_single_test, test_case): test_case 
                    for test_case in test_cases
                }
                
                for future in concurrent.futures.as_completed(future_to_test):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        test_case = future_to_test[future]
                        logger.error(f"Test {test_case.get('id', 'unknown')} failed with exception: {str(e)}")
                        
                        # Create error result
                        error_result = TestResult(
                            id=test_case.get('id', 'unknown'),
                            q_type=test_case.get('q_type', 'unknown'),
                            user_input=test_case.get('user_input', ''),
                            expected_behavior=test_case.get('expected_behavior', ''),
                            response=None,
                            status='error',
                            error_message=str(e),
                            response_time=0.0
                        )
                        results.append(error_result)
        else:
            # Execute tests sequentially
            for test_case in test_cases:
                result = self._run_single_test(test_case)
                results.append(result)
        
        total_time = time.time() - start_time
        
        # Sort results by test ID to maintain order
        results.sort(key=lambda x: x.id)
        
        # Log summary
        successful = len([r for r in results if r.status == 'success'])
        failed = len(results) - successful
        
        logger.info(f"Test execution completed in {total_time:.2f}s")
        logger.info(f"Results: {successful} successful, {failed} failed")
        
        return results
    
    def save_results(self, results: List[TestResult], output_file: str) -> None:
        """
        Save test results to a JSONL file.
        
        Args:
            results (List[TestResult]): Test results to save
            output_file (str): Path to output file
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                for result in results:
                    result_dict = {
                        'id': result.id,
                        'q_type': result.q_type,
                        'user_input': result.user_input,
                        'expected_behavior': result.expected_behavior,
                        'response': result.response,
                        'status': result.status,
                        'error_message': result.error_message,
                        'response_time': result.response_time,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'conversation_history': result.conversation_history
                    }
                    
                    json_line = json.dumps(result_dict, ensure_ascii=False)
                    file.write(json_line + '\n')
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")
            raise


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of test case dictionaries
    """
    test_cases = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    test_case = json.loads(line)
                    test_cases.append(test_case)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
        
    except FileNotFoundError:
        logger.error(f"Test cases file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading test cases from {file_path}: {str(e)}")
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


def main():
    """Main function to run tests from command line."""
    parser = argparse.ArgumentParser(description='Run AI chatbot tests against backend API')
    parser.add_argument(
        '--tests', '-t',
        required=True,
        help='Path to JSONL file containing test cases'
    )
    parser.add_argument(
        '--paper', '-p',
        required=True,
        help='Path to text file containing research paper content'
    )
    parser.add_argument(
        '--endpoint', '-e',
        required=True,
        help='Backend API endpoint URL (e.g., http://localhost:8000)'
    )
    parser.add_argument(
        '--output', '-o',
        default='../data/raw_results.jsonl',
        help='Path to output JSONL file (default: ../data/raw_results.jsonl)'
    )
    parser.add_argument(
        '--api-key', '-k',
        help='API key for authentication (can also use CHATBOT_API_KEY env var)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Maximum number of retry attempts (default: 3)'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=5,
        help='Maximum concurrent requests (default: 5, use 1 for sequential)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('CHATBOT_API_KEY')
    if not api_key:
        logger.warning("No API key provided. Proceeding without authentication.")
    
    try:
        # Load test cases and paper content
        test_cases = load_test_cases(args.tests)
        paper_content = load_paper_content(args.paper)
        
        # Initialize test runner
        runner = APITestRunner(
            api_endpoint=args.endpoint,
            paper_content=paper_content,
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.retries,
            concurrent_requests=args.concurrent
        )
        
        # Run tests
        results = runner.run_tests(test_cases, use_concurrent=args.concurrent > 1)
        
        # Save results
        runner.save_results(results, args.output)
        
        # Print summary
        successful = len([r for r in results if r.status == 'success'])
        failed = len(results) - successful
        
        print(f"✅ Test execution completed!")
        print(f"📊 Results: {successful}/{len(results)} successful")
        print(f"📁 Output saved to: {args.output}")
        
        if failed > 0:
            print(f"⚠️  {failed} tests failed")
            return 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
