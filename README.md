# AI-Finance-Tool
An intelligent financial management system built with FastAPI, ChromaDB, and PyTorch. This project goes beyond simple tracking by utilizing Machine Learning to categorize bank statements and Vector Embeddings for semantic search.
🌟 Key Features
AI Batch Import: Seamlessly upload and process CSV bank statements (optimized for Westpac/CommonBank formats).

Multi-Tier Classification:

Rule-based: Instant labeling using regex patterns for known merchants.

Vector Expansion: Propagates labels to similar transactions using vector similarity.

MLP Classifier: A PyTorch-based Neural Network that predicts categories for complex, unseen transactions.

Semantic Search: Powered by the all-MiniLM-L6-v2 model. Search for "Groceries" to find "Woolworths" or "Aldi," even without exact keyword matches.

Real-time Visualization: Dynamic spending breakdown using Chart.js.

🛠️ Technical Stack
Backend: Python, FastAPI, Pandas

AI/ML: PyTorch (Multi-Layer Perceptron), Sentence-Transformers, Scikit-learn

Vector Database: ChromaDB

Frontend: Vanilla JavaScript, CSS3 (Modern Grid/Flexbox), Chart.js

🚀 Getting Started
Clone & Setup:

Bash
git clone https://github.com/your-username/AI-Finance-Tool.git
cd AI-Finance-Tool
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_gh.txt
Run the Backend:

Bash
uvicorn main:app --reload
Launch Frontend:
Open index.html directly in your browser.

This project uses a 3-phase training pipeline to align bank statement abbreviations (like 'rnt') with semantic queries ('rent').

Bash
# Build the image
docker build -t budget-ai .

# Run the container (with local DB persistence)
docker run -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db budget-ai

📂 Project Architecture
main.py: REST API endpoints and core application logic.

importer.py: Handles vector store operations and AI-driven classification.

app.js: Frontend state management and asynchronous data fetching.

model.pth: Pre-trained weights for the MLP categorization model.
