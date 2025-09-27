#!/bin/bash
# AI Chatbot Testing Automation Setup Script
# This script sets up the testing environment and runs a quick validation

set -e  # Exit on any error

echo "🤖 AI Chatbot Testing Automation Setup"
echo "======================================"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if we have Python 3.9+
python3 -c "
import sys
if sys.version_info < (3, 9):
    print('❌ Python 3.9 or higher is required')
    sys.exit(1)
else:
    print('✅ Python version is compatible')
"

# Install required packages
echo "📦 Installing required packages..."
pip3 install requests --user

# Create virtual environment (optional)
read -p "🤔 Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install requests
    echo "✅ Virtual environment created and activated"
    echo "   To activate later: source venv/bin/activate"
fi

# Validate script syntax
echo "🔍 Validating script syntax..."
python3 -m py_compile scripts/generate_questions.py
python3 -m py_compile scripts/run_tests.py  
python3 -m py_compile scripts/judge.py
python3 -m py_compile scripts/report.py
echo "✅ All scripts have valid syntax"

# Check example data
echo "📄 Checking example data..."
if [ -f "data/sample_paper.txt" ]; then
    echo "✅ Sample paper found"
else
    echo "❌ Sample paper not found"
fi

if [ -f "data/test_data.jsonl" ]; then
    echo "✅ Example test data found"
else
    echo "❌ Example test data not found"
fi

# Set up environment variables template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# AI Chatbot Testing Configuration
CHATBOT_API_KEY=your-api-key-here
CHATBOT_ENDPOINT=http://localhost:8000
EOF
    echo "✅ .env template created - please edit with your API details"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
chmod +x scripts/*.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API key and endpoint"
echo "2. Run a test: cd scripts && python3 generate_questions.py --help"
echo "3. See README.md for complete usage instructions"
echo ""
echo "🚀 Quick test command:"
echo "   cd scripts"
echo "   python3 generate_questions.py -i ../data/sample_paper.txt -o ../data/my_tests.jsonl"
echo ""
echo "Happy testing! 🤖"