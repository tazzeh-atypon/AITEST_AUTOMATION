# AI Chatbot Testing Automation System

A comprehensive automation testing system for evaluating AI chatbots that use the Gemini API via backend endpoints. This system generates test questions, executes them against your chatbot API, evaluates responses, and produces detailed reports.

## ⚡ TL;DR - Quick Test

**Want to test immediately? Run this:**

```bash
# Install dependencies and run demo
pip install -r requirements.txt
python3 test_installation.py  # Optional: verify setup
python3 demo_both_judges.py
```

**This works without any API keys and shows you:**
- ✅ Rule-based evaluation (free & fast)
- ✅ Sample test data and reports
- ✅ Complete workflow demonstration

**Got 30 seconds? Here's your result preview:**
- 📊 Average scores across 4 metrics (Relevancy, Adequacy, Clarity, Consistency)
- 📈 Comparison between evaluation approaches
- 📁 Generated reports in `reports/` folder

## 🚀 Features

- **Multi-Type Question Generation**: Generates fact, inference, formatting, history, and unanswerable questions
- **Robust API Testing**: Handles network errors, timeouts, and malformed responses 
- **4-Metric Evaluation**: Evaluates responses on Relevancy, Adequacy, Clarity, and Consistency
- **Comprehensive Reporting**: Produces CSV, HTML, and JSON reports with visualizations
- **CI/CD Ready**: Includes GitHub Actions workflow for automated testing
- **Concurrent Execution**: Supports parallel test execution for faster results

## 📁 Project Structure

```
AITEST_AUTOMATION/
├── scripts/
│   ├── generate_questions.py    # Generate test questions from research papers
│   ├── run_tests.py            # Execute tests against backend API
│   ├── judge.py                # Evaluate chatbot responses  
│   └── report.py               # Generate comprehensive reports
├── data/
│   ├── sample_paper.txt        # Example research paper content
│   ├── test_data.jsonl         # Generated test questions (JSONL format)
│   ├── raw_results.jsonl       # API responses from test execution
│   └── evaluated.jsonl         # Evaluation results with scores
├── reports/                    # Generated reports (CSV, HTML, JSON)
├── .github/workflows/          # GitHub Actions CI/CD configuration
└── README.md                   # This file
```

## 🚀 Quick Start Testing Guide

**Get your system tested in 5 minutes!**

### Step 1: Setup Environment
```bash
# 1. Clone/navigate to the project
cd AITEST_AUTOMATION

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template and add your API keys
cp .env.example .env
# Edit .env file with your actual API keys (optional for demo)
```

### Step 2: Run Complete Demo
```bash
# Run the full demo (works with or without API keys)
python3 demo_both_judges.py
```

**That's it!** The demo will:
- ✅ Use included sample data if no API keys provided
- ✅ Show rule-based evaluation (always works)
- ✅ Show OpenAI evaluation (if API key provided)
- ✅ Generate comparison reports

### Step 3: Manual Testing (Optional)
```bash
# 1. Generate questions from research paper
python3 scripts/generate_questions.py -i data/sample_paper.txt -o data/my_questions.jsonl

# 2. Run tests against your API (replace with your endpoint)
python3 scripts/run_tests.py \
    -t data/test_data.jsonl \
    -p data/sample_paper.txt \
    -e http://localhost:8000 \
    -o data/my_results.jsonl

# 3. Evaluate responses 
python3 scripts/judge.py \
    -r data/my_results.jsonl \
    -p data/sample_paper.txt \
    -o data/my_evaluations.jsonl

# 4. Generate reports
python3 scripts/report.py \
    -e data/my_evaluations.jsonl \
    -r data/my_results.jsonl \
    -o reports/
```

### Step 4: View Results
- **HTML Report**: Open `reports/test_report_*.html` in your browser
- **CSV Data**: Use `reports/test_report_*.csv` for analysis
- **JSON Summary**: Use `reports/test_summary_*.json` for CI/CD

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Access to your chatbot's backend API endpoint (optional for demo)

### Install Dependencies

```bash
# Install required Python packages
pip install requests concurrent-futures dataclasses pathlib argparse logging csv json datetime

# Or create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install requests
```

### Environment Configuration

Set your API key as an environment variable:

```bash
export CHATBOT_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
echo "CHATBOT_API_KEY=your-api-key-here" > .env
```

## 🧪 Testing Scenarios

### Scenario 1: Test Without API Keys (Demo Mode)
```bash
# This works immediately without any setup
python3 demo_both_judges.py
```
**Perfect for:** First-time users, demonstrations, CI/CD testing

### Scenario 2: Test with Your Chatbot API
```bash
# Add your API endpoint to .env file
echo "CHATBOT_ENDPOINT=http://your-api.com" >> .env

# Run tests
python3 scripts/run_tests.py -t data/test_data.jsonl -p data/sample_paper.txt -e http://your-api.com
```
**Perfect for:** Production testing, API validation

### Scenario 3: Compare Rule-Based vs OpenAI Evaluation
```bash
# Add OpenAI key to .env
echo "OPENAI_API_KEY=sk-..." >> .env

# Run comparison
python3 demo_both_judges.py
```
**Perfect for:** Evaluation quality analysis, research

### Scenario 4: Custom Question Generation
```bash
# Use your own research paper
python3 scripts/generate_questions.py -i your_paper.txt -o custom_questions.jsonl

# Test with custom questions  
python3 scripts/run_tests.py -t custom_questions.jsonl -p your_paper.txt -e http://your-api.com
```
**Perfect for:** Domain-specific testing, custom content

## ✅ Testing Checklist

**Before you start:**
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] In correct directory (contains `demo_both_judges.py`)

**Quick Demo Test (2 minutes):**
- [ ] Run: `python3 demo_both_judges.py`
- [ ] Check console output for test results
- [ ] Verify files created in `data/` directory

**Production API Test (5 minutes):**
- [ ] Add your API endpoint to `.env` file
- [ ] Run: `python3 scripts/run_tests.py -t data/test_data.jsonl -p data/sample_paper.txt -e YOUR_API`
- [ ] Check `data/raw_results.jsonl` for responses
- [ ] Run evaluation: `python3 scripts/judge.py -r data/raw_results.jsonl -p data/sample_paper.txt`

**Full Evaluation Test (10 minutes):**
- [ ] Add OpenAI API key to `.env` (optional)
- [ ] Run: `python3 demo_both_judges.py`
- [ ] Compare rule-based vs OpenAI evaluation results
- [ ] Open HTML report in `reports/` folder

**Custom Content Test (15 minutes):**
- [ ] Add your research paper as `.txt` file
- [ ] Generate questions: `python3 scripts/generate_questions.py -i your_paper.txt`
- [ ] Run full testing pipeline with your content
- [ ] Generate custom reports

---

## 📚 Usage Guide

### Step 1: Generate Test Questions

```bash
cd scripts
python generate_questions.py \
    --input ../data/sample_paper.txt \
    --output ../data/test_data.jsonl \
    --title "Your Research Paper Title"
```

**Options:**
- `--input, -i`: Path to research paper text file (required)
- `--output, -o`: Output JSONL file path (default: ../data/test_data.jsonl)  
- `--title, -t`: Paper title for context (default: Research Paper)
- `--verbose, -v`: Enable verbose logging

### Step 2: Run Tests Against API

```bash
python run_tests.py \
    --tests ../data/test_data.jsonl \
    --paper ../data/sample_paper.txt \
    --endpoint http://localhost:8000 \
    --output ../data/raw_results.jsonl
```

**Options:**
- `--tests, -t`: Path to test questions JSONL file (required)
- `--paper, -p`: Path to research paper text file (required)
- `--endpoint, -e`: Backend API endpoint URL (required)
- `--output, -o`: Output file for test results (default: ../data/raw_results.jsonl)
- `--api-key, -k`: API key (or use CHATBOT_API_KEY env var)
- `--timeout`: Request timeout in seconds (default: 30)
- `--retries`: Max retry attempts (default: 3)
- `--concurrent`: Max concurrent requests (default: 5)

### Step 3: Evaluate Responses

```bash
python judge.py \
    --results ../data/raw_results.jsonl \
    --paper ../data/sample_paper.txt \
    --output ../data/evaluated.jsonl
```

**Options:**
- `--results, -r`: Path to test results JSONL file (required)
- `--paper, -p`: Path to research paper text file (required)  
- `--output, -o`: Output file for evaluations (default: ../data/evaluated.jsonl)
- `--verbose, -v`: Enable verbose logging

### Step 4: Generate Reports

```bash
python report.py \
    --evaluations ../data/evaluated.jsonl \
    --results ../data/raw_results.jsonl \
    --output-dir ../reports \
    --format all
```

**Options:**
- `--evaluations, -e`: Path to evaluations JSONL file (required)
- `--results, -r`: Path to test results JSONL file (required)
- `--output-dir, -o`: Output directory for reports (default: ../reports)
- `--format`: Report format - csv, html, json, or all (default: all)
- `--verbose, -v`: Enable verbose logging

## 🔄 Complete Workflow Example

Run the entire testing pipeline:

```bash
#!/bin/bash
cd scripts

# Generate questions from research paper
echo "🔍 Generating test questions..."
python generate_questions.py -i ../data/sample_paper.txt -o ../data/test_data.jsonl

# Run tests against chatbot API  
echo "🚀 Running tests against API..."
python run_tests.py \
    -t ../data/test_data.jsonl \
    -p ../data/sample_paper.txt \
    -e http://localhost:8000 \
    -o ../data/raw_results.jsonl

# Evaluate responses
echo "⚖️ Evaluating responses..."
python judge.py \
    -r ../data/raw_results.jsonl \
    -p ../data/sample_paper.txt \
    -o ../data/evaluated.jsonl

# Generate comprehensive reports
echo "📊 Generating reports..."  
python report.py \
    -e ../data/evaluated.jsonl \
    -r ../data/raw_results.jsonl \
    -o ../reports

echo "✅ Testing pipeline completed!"
```

## 📊 Understanding the Results

### Evaluation Metrics (0-5 scale)

- **Relevancy**: How well the response relates to the research paper content
- **Adequacy**: Whether the response is complete and sufficient  
- **Clarity**: How clear and understandable the response is
- **Consistency**: Internal logical coherence and consistency

### Question Types

1. **fact**: Direct factual questions answerable from the text
2. **infer**: Questions requiring inference from multiple text parts
3. **not_answerable**: Questions with no answer in the provided text
4. **formatting**: Requests for text transformation (summarize, list, etc.)
5. **history**: Multi-turn conversations testing context preservation

### Report Outputs

- **CSV Report**: Detailed test results in tabular format
- **HTML Report**: Interactive dashboard with visualizations
- **JSON Summary**: Machine-readable summary for CI/CD integration

## 🔧 API Endpoint Requirements

Your backend API endpoint should:

1. **Accept POST requests** to `/chat` endpoint
2. **Handle JSON payloads** with these fields:
   ```json
   {
     "user_input": "The user's question",
     "paper_content": "Full research paper text",  
     "q_type": "question type (fact, infer, etc.)",
     "test_id": "unique test identifier",
     "conversation_history": [optional array for multi-turn]
   }
   ```
3. **Return JSON responses** with:
   ```json
   {
     "response": "The chatbot's answer",
     "status": "success"
   }
   ```
4. **Support authentication** via `Authorization: Bearer <token>` header (optional)

## 🤖 GitHub Actions CI/CD

The included workflow automatically:
- Generates test questions
- Runs tests against your API
- Evaluates responses  
- Produces reports
- Uploads artifacts

Set these repository secrets:
- `CHATBOT_API_KEY`: Your API key
- `CHATBOT_ENDPOINT`: Your API endpoint URL

## 📋 Example Data Formats

### Test Questions (test_data.jsonl)
```json
{"id": "fact_001", "q_type": "fact", "user_input": "What is MedBERT?", "expected_behavior": "Should provide factual answer about MedBERT..."}
```

### Test Results (raw_results.jsonl)  
```json
{"id": "fact_001", "response": "MedBERT is a fine-tuned BERT variant...", "status": "success", "response_time": 1.23}
```

### Evaluations (evaluated.jsonl)
```json
{"id": "fact_001", "relevancy": 5, "adequacy": 4, "clarity": 5, "confusion": 5, "comment": "High quality response..."}
```

## 🚨 Troubleshooting

### Installation Test

**🧪 Run this first to check your setup:**
```bash
python3 test_installation.py
```

This will verify:
- ✅ Python version compatibility
- ✅ All required files present
- ✅ Script syntax validation
- ✅ Dependencies available
- ✅ Data files valid
- ✅ Permissions correct

### Quick Testing Issues

**❓ "Demo doesn't work":**
```bash
# Check if you're in the right directory
ls -la demo_both_judges.py

# Run with verbose output
python3 demo_both_judges.py -v
```

**❓ "No API key found":**
```bash
# Check if .env file exists
ls -la .env

# Create .env if missing
cp .env.example .env
# Edit .env with your API keys

# Verify keys are loaded
python3 -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Gemini:', bool(os.getenv('GEMINI_API_KEY')))"
```

**❓ "Scripts not found":**
```bash
# Make sure you're using python3 not python
which python3

# Check script permissions
ls -la scripts/

# Make scripts executable if needed
chmod +x scripts/*.py
```

**❓ "Import errors":**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install requests openai google-genai
```

**❓ "No test data":**
```bash
# Check if sample data exists
ls -la data/

# Generate test data if missing
python3 scripts/generate_questions.py -i data/sample_paper.txt -o data/test_data.jsonl
```

### Common Issues

**Connection Errors:**
```bash
# Check if API endpoint is accessible
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_input":"test"}'
```

**API Key Issues:**
```bash
# Verify environment variable
echo $CHATBOT_API_KEY

# Check if key is being sent
python -c "import os; print('Key found!' if os.getenv('CHATBOT_API_KEY') else 'No API key')"
```

**JSON Parsing Errors:**
- Ensure API returns valid JSON format
- Check response structure matches expected format
- Enable verbose logging with `-v` flag

**High Failure Rates:**
- Reduce concurrent requests with `--concurrent 1`
- Increase timeout with `--timeout 60`  
- Check API rate limits and quotas

### Debug Mode

Enable verbose logging for all scripts:
```bash
python script_name.py --verbose [other options]
```

### Performance Tuning

**For faster execution:**
- Increase `--concurrent` (but watch API rate limits)
- Reduce `--timeout` if API is consistently fast
- Use `--retries 1` if API is reliable

**For more reliable execution:**
- Set `--concurrent 1` for sequential processing
- Increase `--timeout` for slow responses
- Increase `--retries` for unreliable networks

## 📈 Advanced Usage

### Custom Question Generation

Modify `generate_questions.py` to add new question types:

```python
def generate_custom_questions(self, count: int = 3) -> List[Dict[str, Any]]:
    # Your custom question generation logic
    questions = []
    # ... implementation
    return questions
```

### Custom Evaluation Metrics

Extend `judge.py` to add new evaluation dimensions:

```python
def _evaluate_custom_metric(self, response: str) -> Tuple[int, str]:
    # Your custom evaluation logic
    score = 0  # 0-5 scale
    explanation = "Custom metric evaluation"
    return score, explanation
```

### Integration with Other Tools

The system produces standard formats that integrate with:
- **Jenkins**: Use JSON summaries for build status
- **Grafana**: Import CSV data for monitoring dashboards  
- **Slack**: Parse HTML reports for team notifications
- **Jira**: Create issues from failing test reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

### Development Guidelines

- Follow PEP8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write tests for new functionality
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:

1. **Check the troubleshooting section** above
2. **Review example data formats** to ensure compatibility
3. **Enable verbose logging** to debug issues
4. **Open an issue** with detailed logs and error messages

## 🔮 Roadmap

- [ ] Support for additional question types
- [ ] Integration with more evaluation frameworks  
- [ ] Real-time monitoring dashboard
- [ ] Multi-language support
- [ ] Advanced bias detection metrics
- [ ] Integration with popular chatbot frameworks

---

**Happy Testing! 🚀**

For the most up-to-date information, please check the repository and documentation.




# 1. Set up environment
./setup.sh
export CHATBOT_API_KEY="your-key"

# 2. Generate questions
cd scripts
python3 generate_questions.py -i ../data/sample_paper.txt -o ../data/test_data.jsonl

# 3. Run tests
python3 run_tests.py -t ../data/test_data.jsonl -p ../data/sample_paper.txt -e http://localhost:8000

# 4. Evaluate responses  
python3 judge.py -r ../data/raw_results.jsonl -p ../data/sample_paper.txt

# 5. Generate reports
python3 report.py -e ../data/evaluated.jsonl -r ../data/raw_results.jsonl