# 🤖 TrioPost: Multi-Agent AI Social Commerce Dashboard

TrioPost is an automated AI content creator designed specifically for social commerce. By leveraging a Multi-Agent System and State Graph Workflow, it solves the challenges of writing marketing copy, researching market prices, and maintaining consistent brand tone. The system autonomously orchestrates multiple AI agents to synthesize high-converting, platform-optimized posts from a single product image.
## ✨ Key Features

* **Multi-Agent Orchestration:** Coordinates specialized AI agents (Vision, Research, RAG, Copywriter) seamlessly using LangGraph.
* **Brand Voice Cloning (RAG):** Utilizes a Vector Database and Few-Shot Prompting to inject specific brand guidelines and tones into the AI's responses.
* **Dynamic Research:** Automatically fetches live product specs, reviews, and market data via the Tavily Search API to ground the AI's content in real-world facts.
* **Platform-Specific Optimization:** Dynamically adjusts content structure, tone, and hashtag density for Facebook (Storytelling), Instagram (Aesthetic & Hashtag-heavy), and X/Twitter (Concise under 280 chars).
* **Human-in-the-Loop & Reflection Mechanism:** Users can provide feedback to rewrite captions. Instead of a basic zero-shot rewrite, the system uses a Critic Agent to formulate a revision plan before the Writer Agent generates the final output, ensuring maximum accuracy and preventing AI stubbornness.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend / UI** | Streamlit |
| **AI Orchestration** | LangChain, LangGraph |
| **Core LLM & Vision** | Typhoon API (v2.5 & OCR) |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace (`sentence-transformers`) |
| **Web Search Tool** | Tavily API |

## 🧠 The Agentic Workflow (Data Pipeline)

TrioPost moves away from traditional linear prompting. It uses a graph-based state machine (`TrioPostState`) to pass context between specialized nodes:
1. **The Vision Agent:** Acts as the "Eyes". It scans the uploaded product image to extract distinct visual features, colors, and materials while strictly preventing feature hallucination.
2. **The Researcher Agent:** Acts as the "Detective". It takes the visual context (or user-defined product name) and searches the web for live market specs, reviews, and reference data.
3. **The RAG Node:** Acts as the "Memory". It queries ChromaDB to retrieve specific writing guidelines and past post examples based on the selected store.
4. **The Writer Agent:** Acts as the "Master Writer". It synthesizes the visual data, internet research, brand rules, user constraints, and platform formats to generate the final marketing caption.

## 🚀 Getting Started
To run the TrioPost dashboard locally:

### 1. Prerequisites
* Python 3.9+
* API Keys for [OpenTyphoon](https://opentyphoon.ai/) and [Tavily](https://www.tavily.com/)

### 2. Environment Variables
Create a `.env` file in the root directory of your project:

```env
TYPHOON_API_KEY=your_typhoon_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Installation & Run
1. Clone the repository and navigate to the project folder.
2. Set up your Python virtual environment and install dependencies:
    ```bash
    python -m venv venv
    # Activate: venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/macOS)
    pip install -r requirements.txt
    ```
3. Initialize the Vector Database (Add sample stores to ChromaDB):
   (Run the RAG manager page or your specific DB initialization script first to populate the knowledge base)
4. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

**The dashboard will be accessible in your web browser at** `http://localhost:8501/`
