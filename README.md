# Scientific Research Assistant Agent

A multi-agent research assistant system for analyzing arXiv papers in the cs.LG category, generating hypotheses, and deriving insights using the Agent Development Kit (ADK).

## 🎯 Project Overview

This project implements a sophisticated research assistant that:
- Analyzes 10,000+ arXiv papers in the cs.LG category
- Generates research hypotheses and corresponding code
- Derives meaningful trends using BigQuery
- Presents insights with visualizations and reports
- Ensures human-in-the-loop quality control at each step

## 🏗️ Architecture

The system consists of seven specialized agents:

1. **SearchAgent**: Vector-based paper search and retrieval
2. **HypothesisAgent**: Research hypothesis generation and refinement
3. **InsightAgent**: Trend analysis and pattern discovery
4. **CodeAgent**: Code generation and validation
5. **VisualizationAgent**: Data visualization and chart generation
6. **ReportAgent**: Research report compilation and citation management
7. **NoteTaker**: System-wide logging and feedback tracking

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB Atlas account
- Google Cloud account with BigQuery enabled
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone [YOUR_REPO_URL]
   cd scientific-research-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.template .env
   ```
   Edit `.env` with your credentials:
   - CHATGPT_API_KEY
   - MONGO_URI
   - GOOGLE_CLOUD_PROJECT

### Running the Application

1. Start the backend server:
   ```bash
   python src/main.py
   ```

2. Start the frontend development server:
   ```bash
   cd src/frontend
   npm install
   npm start
   ```

## 🧪 Development

### Code Structure

```
src/
├── agents/           # Agent implementations
├── utils/           # Utility functions
├── config/          # Configuration files
└── frontend/        # React frontend
```

### Testing

Run the test suite:
```bash
pytest tests/
```

### Code Quality

The project uses pylint for code quality:
```bash
pylint src/
```

## 📝 Citation Logic

The system uses arXiv metadata for citations with the following format:
```
Author, A., & Author, B. (Year). Title. arXiv:XXXX.XXXXX
```

If Google Scholar API is accessible, citation counts are augmented.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Agent Development Kit (ADK) team
- arXiv for providing the paper dataset
- Google Cloud Platform for infrastructure support 