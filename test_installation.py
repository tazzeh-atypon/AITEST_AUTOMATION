#!/usr/bin/env python3
"""
Installation and Setup Test Script

This script verifies that the testing system is properly set up and ready to use.
Run this after following the README setup instructions.

Author: AI Testing System  
Date: 2025-09-27
"""

import os
import sys
import subprocess
import json

def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 9):
        print("✅ Python version is compatible (3.9+)")
        return True
    else:
        print("❌ Python 3.9+ required")
        return False

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        'scripts/generate_questions.py',
        'scripts/run_tests.py', 
        'scripts/judge.py',
        'scripts/report.py',
        'demo_both_judges.py',
        'data/sample_paper.txt',
        'data/test_data.jsonl',
        'README.md',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_script_syntax():
    """Test that all Python scripts have valid syntax."""
    script_files = [
        'scripts/generate_questions.py',
        'scripts/run_tests.py',
        'scripts/judge.py', 
        'scripts/report.py',
        'scripts/simple_openai_judge.py',
        'demo_both_judges.py'
    ]
    
    syntax_errors = []
    for script in script_files:
        if os.path.exists(script):
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', script], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Syntax OK: {script}")
                else:
                    print(f"❌ Syntax Error: {script}")
                    print(f"   Error: {result.stderr}")
                    syntax_errors.append(script)
            except Exception as e:
                print(f"❌ Cannot test: {script} - {str(e)}")
                syntax_errors.append(script)
    
    return len(syntax_errors) == 0

def test_dependencies():
    """Test that required dependencies are available."""
    dependencies = {
        'requests': 'HTTP requests library',
        'json': 'JSON parsing (built-in)',
        'os': 'Operating system interface (built-in)',
        'argparse': 'Command line argument parsing (built-in)'
    }
    
    missing_deps = []
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ Import OK: {dep} - {description}")
        except ImportError:
            print(f"❌ Missing: {dep} - {description}")
            missing_deps.append(dep)
    
    # Test optional dependencies
    optional_deps = {
        'openai': 'OpenAI API client (optional)',
        'google.genai': 'Google Gemini API client (optional)'
    }
    
    for dep, description in optional_deps.items():
        try:
            if '.' in dep:
                parts = dep.split('.')
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(dep)
            print(f"✅ Optional OK: {dep} - {description}")
        except (ImportError, AttributeError):
            print(f"⚠️  Optional Missing: {dep} - {description}")
    
    return len(missing_deps) == 0

def test_data_files():
    """Test that data files are valid."""
    data_tests = []
    
    # Test sample paper
    if os.path.exists('data/sample_paper.txt'):
        try:
            with open('data/sample_paper.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > 1000:
                print("✅ Sample paper: Valid content")
                data_tests.append(True)
            else:
                print("❌ Sample paper: Content too short")
                data_tests.append(False)
        except Exception as e:
            print(f"❌ Sample paper: Read error - {str(e)}")
            data_tests.append(False)
    
    # Test sample questions
    if os.path.exists('data/test_data.jsonl'):
        try:
            with open('data/test_data.jsonl', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            valid_json_count = 0
            for line_num, line in enumerate(lines, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'id' in data and 'user_input' in data:
                            valid_json_count += 1
                    except json.JSONDecodeError:
                        print(f"❌ Test data line {line_num}: Invalid JSON")
            
            if valid_json_count > 10:
                print(f"✅ Test data: {valid_json_count} valid questions")
                data_tests.append(True)
            else:
                print(f"❌ Test data: Only {valid_json_count} valid questions")
                data_tests.append(False)
                
        except Exception as e:
            print(f"❌ Test data: Read error - {str(e)}")
            data_tests.append(False)
    
    return all(data_tests)

def test_permissions():
    """Test that scripts have proper permissions."""
    script_files = [
        'scripts/generate_questions.py',
        'scripts/run_tests.py',
        'scripts/judge.py',
        'scripts/report.py',
        'demo_both_judges.py',
        'setup.sh'
    ]
    
    permission_issues = []
    for script in script_files:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✅ Executable: {script}")
            else:
                print(f"⚠️  Not executable: {script} (will still work with python3)")
                # This is not a critical error on all systems
        else:
            permission_issues.append(script)
    
    return len(permission_issues) == 0

def run_all_tests():
    """Run all installation tests."""
    print("🧪 AI Chatbot Testing System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("File Structure", test_file_structure), 
        ("Script Syntax", test_script_syntax),
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_files),
        ("Permissions", test_permissions)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready for testing. You can now run:")
        print("   python3 demo_both_judges.py")
    elif passed >= len(results) - 1:
        print("\n⚠️  MOSTLY READY!")
        print("One minor issue detected, but the system should still work.")
        print("Try running: python3 demo_both_judges.py")
    else:
        print("\n🔧 SETUP NEEDED!")
        print("Please fix the failed tests before proceeding.")
        print("Check the README.md for setup instructions.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
