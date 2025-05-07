# Advanced ATS Resume Scoring System

<div align="center">

![ATS Resume Analyzer](https://img.shields.io/badge/ATS-Resume%20Analyzer-blue)
![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)
![NLP](https://img.shields.io/badge/NLP-Powered-orange)

</div>

## 📋 Overview

This application uses advanced NLP techniques and Large Language Models to analyze resumes against job descriptions, providing a comprehensive scoring system with detailed explanations for better job applications.

##  Features

- **🔍 Technology Extraction**: Uses LLM to extract technologies mentioned in resumes and job descriptions
- **🧩 Skills Matching**: Implements asymmetric Jaccard similarity to prioritize matches from required skills
- **📊 Experience Analysis**: Extracts and compares years of experience with penalties for missing years
- **🔤 Semantic Content Matching**: Uses Sentence-BERT (SBERT) embeddings and cosine similarity to compare resume and job summaries
- **⭐ Comprehensive Scoring**: Combines multiple factors with weighted scoring
- **📈 Visual Explanations**: Provides detailed breakdowns of match scores with visual aids
- **💡 Actionable Recommendations**: Suggests improvements based on analysis

##  Installation

1. Clone the repository
   ```bash
   git clone https://github.com/karim12348585/ATS-Resume-Scoring-System
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Poppler for PDF to image conversion:
   - **Ubuntu/Debian**: `apt-get install poppler-utils`
   - **macOS**: `brew install poppler`
   - **Windows**: Download and install from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

4. Create a `.env` file in the project root with your API key:
   ```
   api_key = your_gemini_api_key_here
   ```

##  Usage

1. Run the Streamlit app:
   ```bash
   streamlit run enhanced_ats.py
   ```
2. Upload your resume (PDF format)
3. Enter the job description text
4. Click "Analyze Resume Match" to get your results

## 📊 How the Scoring Works

The final score is calculated as a weighted average of three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Skills Match** | 50% | Using asymmetric Jaccard similarity to prioritize required skills |
| **Experience Match** | 30% | Comparing years of experience with penalties for missing years |
| **Profile Alignment** | 20% | Using SBERT and cosine similarity for semantic similarity |

##  System Architecture

```
                           ┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
                           │  Resume (PDF) │     │ Job Description  │     │  SBERT Model    │
                           └───────┬───────┘     └────────┬─────────┘     └────────┬────────┘
                                   │                      │                        │
                                   ▼                      ▼                        ▼
                           ┌─────────────────────────────────────────────────────────────────┐
                           │                    Data Extraction Layer                        │
                           │  - Extract technologies using LLM                               │
                           │  - Extract experience years                                     │
                           │  - Generate text summaries                                      │
                           └───────────────────────────────┬─────────────────────────────────┘
                                                           │
                                                           ▼
                           ┌─────────────────────────────────────────────────────────────────┐
                           │                    Analysis Layer                               │
                           │  - Asymmetric Jaccard for skills                                │
                           │  - Experience comparison with penalties                         │
                           │  - SBERT embeddings and cosine similarity                       │
                           └───────────────────────────────┬─────────────────────────────────┘
                                                           │
                                                           ▼
                           ┌─────────────────────────────────────────────────────────────────┐
                           │                    Scoring Layer                                │
                           │  - Weighted average of component scores                         │
                           │  - Explanation generation                                       │
                           │  - Recommendations                                              │
                           └───────────────────────────────┬─────────────────────────────────┘
                                                           │
                                                           ▼
                           ┌─────────────────────────────────────────────────────────────────┐
                           │                    Presentation Layer                           │
                           │  - Interactive Streamlit UI                                     │
                           │  - Visual score representations                                 │
                           │  - Detailed analytics tabs                                      │
                           └─────────────────────────────────────────────────────────────────┘
```

##  Screenshots

<div align="center">
  <img src="/api/placeholder/800/400" alt="Application Screenshot" />
  <p><i>Sample analysis with score breakdown</i></p>
</div>

## 📝 Todo

- [ ] Add support for more resume formats (DOCX, HTML)
- [ ] Implement custom weighting for different job roles
- [ ] Create API endpoints for integration with other tools
- [ ] Add benchmarking against successful resumes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


