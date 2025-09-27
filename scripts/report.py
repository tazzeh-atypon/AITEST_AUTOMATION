#!/usr/bin/env python3
"""
Report Generation Script for AI Chatbot Testing

This script generates comprehensive reports from evaluation results,
including CSV summaries and HTML reports with visualizations.

Author: AI Testing System
Date: 2025-09-27
"""

import json
import argparse
import logging
import os
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestSummary:
    """Data class for storing test summary statistics."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    error_tests: int
    avg_relevancy: float
    avg_adequacy: float
    avg_clarity: float
    avg_confusion: float
    avg_overall: float
    test_types: Dict[str, int]
    response_types: Dict[str, int]


@dataclass
class TestDetail:
    """Data class for detailed test information."""
    id: str
    q_type: str
    user_input: str
    response: str
    status: str
    relevancy: int
    adequacy: int
    clarity: int
    confusion: int
    overall: float
    comment: str
    response_type: str
    response_time: float


class ReportGenerator:
    """
    Generates comprehensive reports from chatbot evaluation results.
    
    Supports multiple output formats including CSV, HTML, and JSON summaries
    with detailed analytics and visualization capabilities.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    def load_evaluations(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load evaluation results from a JSONL file.
        
        Args:
            file_path (str): Path to the evaluation JSONL file
            
        Returns:
            List[Dict]: List of evaluation result dictionaries
        """
        evaluations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        evaluation = json.loads(line)
                        evaluations.append(evaluation)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(evaluations)} evaluations from {file_path}")
            return evaluations
            
        except FileNotFoundError:
            logger.error(f"Evaluations file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading evaluations from {file_path}: {str(e)}")
            raise
    
    def load_test_results(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load test results from a JSONL file.
        
        Args:
            file_path (str): Path to the test results JSONL file
            
        Returns:
            List[Dict]: List of test result dictionaries
        """
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(results)} test results from {file_path}")
            return results
            
        except FileNotFoundError:
            logger.error(f"Test results file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading test results from {file_path}: {str(e)}")
            raise
    
    def merge_data(self, evaluations: List[Dict[str, Any]], 
                   test_results: List[Dict[str, Any]]) -> List[TestDetail]:
        """
        Merge evaluation results with test results to create complete test details.
        
        Args:
            evaluations (List[Dict]): Evaluation results
            test_results (List[Dict]): Test execution results
            
        Returns:
            List[TestDetail]: Combined test details
        """
        # Create lookup dictionary for test results
        results_lookup = {result['id']: result for result in test_results}
        
        test_details = []
        
        for evaluation in evaluations:
            test_id = evaluation.get('id', 'unknown')
            test_result = results_lookup.get(test_id, {})
            
            # Calculate overall score
            scores = [
                evaluation.get('relevancy', 0),
                evaluation.get('adequacy', 0),
                evaluation.get('clarity', 0),
                evaluation.get('confusion', 0)
            ]
            overall_score = sum(scores) / len(scores)
            
            detail = TestDetail(
                id=test_id,
                q_type=test_result.get('q_type', 'unknown'),
                user_input=test_result.get('user_input', ''),
                response=test_result.get('response', ''),
                status=test_result.get('status', 'unknown'),
                relevancy=evaluation.get('relevancy', 0),
                adequacy=evaluation.get('adequacy', 0),
                clarity=evaluation.get('clarity', 0),
                confusion=evaluation.get('confusion', 0),
                overall=overall_score,
                comment=evaluation.get('comment', ''),
                response_type=evaluation.get('response_type', 'unknown'),
                response_time=test_result.get('response_time', 0.0)
            )
            
            test_details.append(detail)
        
        logger.info(f"Merged data for {len(test_details)} tests")
        return test_details
    
    def calculate_summary(self, test_details: List[TestDetail]) -> TestSummary:
        """
        Calculate summary statistics from test details.
        
        Args:
            test_details (List[TestDetail]): List of test details
            
        Returns:
            TestSummary: Aggregated summary statistics
        """
        if not test_details:
            return TestSummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})
        
        total_tests = len(test_details)
        successful_tests = len([t for t in test_details if t.status == 'success'])
        failed_tests = len([t for t in test_details if t.status in ['error', 'timeout', 'invalid_response']])
        error_tests = total_tests - successful_tests - failed_tests
        
        # Calculate average scores
        avg_relevancy = sum(t.relevancy for t in test_details) / total_tests
        avg_adequacy = sum(t.adequacy for t in test_details) / total_tests
        avg_clarity = sum(t.clarity for t in test_details) / total_tests
        avg_confusion = sum(t.confusion for t in test_details) / total_tests
        avg_overall = sum(t.overall for t in test_details) / total_tests
        
        # Count test types
        test_types = {}
        for test in test_details:
            test_types[test.q_type] = test_types.get(test.q_type, 0) + 1
        
        # Count response types
        response_types = {}
        for test in test_details:
            response_types[test.response_type] = response_types.get(test.response_type, 0) + 1
        
        return TestSummary(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            avg_relevancy=avg_relevancy,
            avg_adequacy=avg_adequacy,
            avg_clarity=avg_clarity,
            avg_confusion=avg_confusion,
            avg_overall=avg_overall,
            test_types=test_types,
            response_types=response_types
        )
    
    def generate_csv_report(self, test_details: List[TestDetail], 
                           summary: TestSummary, output_file: str) -> None:
        """
        Generate a CSV report with detailed test results.
        
        Args:
            test_details (List[TestDetail]): Test details
            summary (TestSummary): Summary statistics
            output_file (str): Output CSV file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'test_id', 'question_type', 'status', 'response_type',
                    'relevancy', 'adequacy', 'clarity', 'confusion', 'overall_score',
                    'response_time', 'user_input', 'response', 'comment'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for test in test_details:
                    writer.writerow({
                        'test_id': test.id,
                        'question_type': test.q_type,
                        'status': test.status,
                        'response_type': test.response_type,
                        'relevancy': test.relevancy,
                        'adequacy': test.adequacy,
                        'clarity': test.clarity,
                        'confusion': test.confusion,
                        'overall_score': f"{test.overall:.2f}",
                        'response_time': f"{test.response_time:.2f}",
                        'user_input': test.user_input[:100] + ('...' if len(test.user_input) > 100 else ''),
                        'response': test.response[:200] + ('...' if len(test.response) > 200 else ''),
                        'comment': test.comment[:300] + ('...' if len(test.comment) > 300 else '')
                    })
            
            logger.info(f"CSV report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {str(e)}")
            raise
    
    def generate_html_report(self, test_details: List[TestDetail], 
                            summary: TestSummary, output_file: str) -> None:
        """
        Generate an HTML report with visualizations and detailed analysis.
        
        Args:
            test_details (List[TestDetail]): Test details
            summary (TestSummary): Summary statistics
            output_file (str): Output HTML file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Generate HTML content
            html_content = self._generate_html_content(test_details, summary)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(html_content)
            
            logger.info(f"HTML report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _generate_html_content(self, test_details: List[TestDetail], 
                              summary: TestSummary) -> str:
        """Generate HTML content for the report."""
        
        # Find failing tests (overall score < 3.0)
        failing_tests = [t for t in test_details if t.overall < 3.0]
        top_performing = sorted(test_details, key=lambda x: x.overall, reverse=True)[:5]
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: white;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .score-bar {{
            background-color: #ddd;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 10px 0;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff4757 0%, #ffa502 50%, #2ed573 100%);
            transition: width 0.3s ease;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .status-success {{ color: #2ed573; font-weight: bold; }}
        .status-error {{ color: #ff4757; font-weight: bold; }}
        .status-timeout {{ color: #ffa502; font-weight: bold; }}
        .test-section {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .section-header {{
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            margin: 0;
        }}
        .section-content {{
            padding: 20px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .bar-chart {{
            display: flex;
            align-items: end;
            height: 200px;
            margin: 20px 0;
            padding: 0 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }}
        .bar {{
            flex: 1;
            background: linear-gradient(180deg, #4CAF50 0%, #2ed573 100%);
            margin: 0 5px;
            border-radius: 4px 4px 0 0;
            display: flex;
            align-items: end;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
            padding: 5px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Chatbot Test Report</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <h3>Total Tests</h3>
                <div class="metric-value">{summary.total_tests}</div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value">{(summary.successful_tests/summary.total_tests*100):.1f}%</div>
            </div>
            <div class="metric-card">
                <h3>Average Score</h3>
                <div class="metric-value">{summary.avg_overall:.2f}/5</div>
            </div>
            <div class="metric-card">
                <h3>Failed Tests</h3>
                <div class="metric-value">{summary.failed_tests}</div>
            </div>
        </div>
        
        <div class="test-section">
            <h2 class="section-header">📊 Performance Metrics</h2>
            <div class="section-content">
                <div class="chart-container">
                    <h3>Average Scores by Metric</h3>
                    <div class="bar-chart">
                        <div class="bar" style="height: {(summary.avg_relevancy/5)*180}px;">
                            Relevancy<br>{summary.avg_relevancy:.1f}
                        </div>
                        <div class="bar" style="height: {(summary.avg_adequacy/5)*180}px;">
                            Adequacy<br>{summary.avg_adequacy:.1f}
                        </div>
                        <div class="bar" style="height: {(summary.avg_clarity/5)*180}px;">
                            Clarity<br>{summary.avg_clarity:.1f}
                        </div>
                        <div class="bar" style="height: {(summary.avg_confusion/5)*180}px;">
                            Consistency<br>{summary.avg_confusion:.1f}
                        </div>
                    </div>
                </div>
                
                <h3>Test Types Distribution</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Question Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for q_type, count in summary.test_types.items():
            percentage = (count / summary.total_tests) * 100
            html += f"""
                        <tr>
                            <td>{q_type}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
"""
        
        if failing_tests:
            html += f"""
        <div class="test-section">
            <h2 class="section-header">⚠️ Failing Tests (Score < 3.0)</h2>
            <div class="section-content">
                <p>Found {len(failing_tests)} tests with scores below the threshold.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Test ID</th>
                            <th>Type</th>
                            <th>Overall Score</th>
                            <th>Issues</th>
                            <th>Question (Preview)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for test in failing_tests[:10]:  # Show top 10 failing tests
                html += f"""
                        <tr>
                            <td>{test.id}</td>
                            <td>{test.q_type}</td>
                            <td>{test.overall:.2f}</td>
                            <td>
                                {'❌ Relevancy' if test.relevancy < 3 else ''}
                                {'❌ Adequacy' if test.adequacy < 3 else ''}
                                {'❌ Clarity' if test.clarity < 3 else ''}
                                {'❌ Consistency' if test.confusion < 3 else ''}
                            </td>
                            <td>{test.user_input[:80]}{'...' if len(test.user_input) > 80 else ''}</td>
                        </tr>
"""
            
            html += """
                    </tbody>
                </table>
            </div>
        </div>
"""
        
        html += f"""
        <div class="test-section">
            <h2 class="section-header">🏆 Top Performing Tests</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Test ID</th>
                            <th>Type</th>
                            <th>Overall Score</th>
                            <th>R</th>
                            <th>A</th>
                            <th>Cl</th>
                            <th>Co</th>
                            <th>Question (Preview)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for test in top_performing:
            html += f"""
                        <tr>
                            <td>{test.id}</td>
                            <td>{test.q_type}</td>
                            <td><strong>{test.overall:.2f}</strong></td>
                            <td>{test.relevancy}</td>
                            <td>{test.adequacy}</td>
                            <td>{test.clarity}</td>
                            <td>{test.confusion}</td>
                            <td>{test.user_input[:80]}{'...' if len(test.user_input) > 80 else ''}</td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="test-section">
            <h2 class="section-header">📋 All Test Results</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Score</th>
                            <th>R</th>
                            <th>A</th>
                            <th>Cl</th>
                            <th>Co</th>
                            <th>Time(s)</th>
                            <th>Question</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for test in sorted(test_details, key=lambda x: x.id):
            status_class = f"status-{test.status}"
            html += f"""
                        <tr>
                            <td>{test.id}</td>
                            <td>{test.q_type}</td>
                            <td class="{status_class}">{test.status}</td>
                            <td><strong>{test.overall:.2f}</strong></td>
                            <td>{test.relevancy}</td>
                            <td>{test.adequacy}</td>
                            <td>{test.clarity}</td>
                            <td>{test.confusion}</td>
                            <td>{test.response_time:.2f}</td>
                            <td>{test.user_input[:60]}{'...' if len(test.user_input) > 60 else ''}</td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def generate_json_summary(self, summary: TestSummary, output_file: str) -> None:
        """
        Generate a JSON summary report suitable for CI/CD integration.
        
        Args:
            summary (TestSummary): Summary statistics
            output_file (str): Output JSON file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            summary_dict = asdict(summary)
            summary_dict['timestamp'] = datetime.datetime.now().isoformat()
            summary_dict['success_rate'] = summary.successful_tests / summary.total_tests if summary.total_tests > 0 else 0
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(summary_dict, file, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON summary saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating JSON summary: {str(e)}")
            raise


def main():
    """Main function to generate reports from command line."""
    parser = argparse.ArgumentParser(description='Generate comprehensive test reports')
    parser.add_argument(
        '--evaluations', '-e',
        required=True,
        help='Path to JSONL file containing evaluations'
    )
    parser.add_argument(
        '--results', '-r',
        required=True,
        help='Path to JSONL file containing test results'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='../reports',
        help='Output directory for reports (default: ../reports)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'html', 'json', 'all'],
        default='all',
        help='Report format to generate (default: all)'
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
        # Initialize report generator
        generator = ReportGenerator()
        
        # Load data
        evaluations = generator.load_evaluations(args.evaluations)
        test_results = generator.load_test_results(args.results)
        
        # Merge and analyze data
        test_details = generator.merge_data(evaluations, test_results)
        summary = generator.calculate_summary(test_details)
        
        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate requested reports
        if args.format in ['csv', 'all']:
            csv_file = os.path.join(args.output_dir, f'test_report_{timestamp}.csv')
            generator.generate_csv_report(test_details, summary, csv_file)
            print(f"📊 CSV report: {csv_file}")
        
        if args.format in ['html', 'all']:
            html_file = os.path.join(args.output_dir, f'test_report_{timestamp}.html')
            generator.generate_html_report(test_details, summary, html_file)
            print(f"🌐 HTML report: {html_file}")
        
        if args.format in ['json', 'all']:
            json_file = os.path.join(args.output_dir, f'test_summary_{timestamp}.json')
            generator.generate_json_summary(summary, json_file)
            print(f"📋 JSON summary: {json_file}")
        
        # Print summary to console
        print(f"\n✅ Report generation completed!")
        print(f"📊 Summary:")
        print(f"   Total Tests: {summary.total_tests}")
        print(f"   Success Rate: {(summary.successful_tests/summary.total_tests*100):.1f}%")
        print(f"   Average Scores:")
        print(f"     Relevancy: {summary.avg_relevancy:.2f}/5")
        print(f"     Adequacy:  {summary.avg_adequacy:.2f}/5")
        print(f"     Clarity:   {summary.avg_clarity:.2f}/5")
        print(f"     Consistency: {summary.avg_confusion:.2f}/5")
        print(f"     Overall:   {summary.avg_overall:.2f}/5")
        
        # Exit with error code if too many tests failed
        if summary.failed_tests > summary.total_tests * 0.1:  # More than 10% failed
            print(f"⚠️  Warning: High failure rate ({summary.failed_tests} failed)")
            return 1
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
