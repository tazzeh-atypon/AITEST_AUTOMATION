#!/usr/bin/env python3
"""
Simple .env file loader

Loads environment variables from .env file so you don't need to export them manually.

Author: AI Testing System
Date: 2025-09-27
"""

import os

def load_env(env_file_path=None):
    """
    Load environment variables from .env file.
    
    Args:
        env_file_path (str): Path to .env file (default: looks for .env in parent directory)
    """
    if env_file_path is None:
        # Look for .env file in current directory or parent
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        env_file_path = os.path.join(parent_dir, '.env')
        if not os.path.exists(env_file_path):
            env_file_path = os.path.join(current_dir, '.env')
    
    if not os.path.exists(env_file_path):
        return  # No .env file found, skip loading
    
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")  # Remove quotes
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        
        print(f"✅ Loaded environment variables from {env_file_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load .env file: {str(e)}")

# Auto-load when module is imported
load_env()
