#!/usr/bin/env python3
"""
Simple Question Generation Script using Google Gemini API

This is a simplified version that uses the Gemini API directly to generate
test questions from research paper content.

Author: AI Testing System
Date: 2025-09-27
"""

import os
import json
import argparse
import logging
from google import genai

# Load environment variables from .env file
from env_loader import load_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini configuration
MODEL = "gemini-2.0-flash"  # or gemini-2.0 depending on availability


def get_genai_client():
    """Initialize and return Gemini client - picks up GEMINI_API_KEY from environment."""
    try:
        return genai.Client()
    except ValueError as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        logger.error("Please set GEMINI_API_KEY environment variable")
        raise


def generate_questions(article_text: str, n: int = 20) -> list:
    """
    Generate test questions using Gemini API.
    
    Args:
        article_text (str): Full text of the research paper
        n (int): Number of questions to generate
        
    Returns:
        list: List of generated questions
    """
    # Create a comprehensive prompt for question generation
    prompt = f"""
    You are an expert test question generator for AI chatbot evaluation. 
    Generate {n} diverse test questions based on the following research paper.
    
    Generate questions of these types:
    1. FACT: Direct factual questions answerable from the text
    2. INFER: Questions requiring inference from multiple text parts  
    3. NOT_ANSWERABLE: Questions that cannot be answered from this text
    4. FORMATTING: Requests for text transformation (summarize, list, etc.)
    5. HISTORY: Multi-turn conversation questions
    
    For each question, provide:
    - id: unique identifier (e.g., "fact_001", "infer_001")
    - q_type: one of the types above
    - user_input: the actual question text
    - expected_behavior: description of expected response
    
    Return ONLY a valid JSON array of question objects.
    
    Article text: {article_text[:3000]}...
    """
    
    try:
        # Initialize client when needed
        genai_client = get_genai_client()
        
        resp = genai_client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        
        # Extract text correctly according to SDK
        text = resp.text
        
        # Attempt to convert resulting JSON to objects
        questions = json.loads(text)
        
        logger.info(f"Generated {len(questions)} questions using Gemini API")
        return questions
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from Gemini: {str(e)}")
        logger.debug(f"Raw response: {text}")
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate questions using Gemini API')
    parser.add_argument(
        '--input', '-i',
        default='article.txt',
        help='Input article file (default: article.txt)'
    )
    parser.add_argument(
        '--output', '-o', 
        default='test_data.jsonl',
        help='Output JSONL file (default: test_data.jsonl)'
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=20,
        help='Number of questions to generate (default: 20)'
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
        # Read article content
        logger.info(f"Reading article from {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            article = f.read()
        
        # Generate questions
        logger.info(f"Generating {args.count} questions...")
        questions = generate_questions(article, n=args.count)
        
        # Save to JSONL format
        logger.info(f"Saving questions to {args.output}")
        with open(args.output, "w", encoding="utf-8") as fout:
            for q in questions:
                record = {"context": article, **q}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"✅ Saved {args.output} with {len(questions)} questions")
        
    except FileNotFoundError:
        logger.error(f"Article file not found: {args.input}")
        return 1
    except Exception as e:
        logger.error(f"Failed to generate questions: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
