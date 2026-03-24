# Multi-Modal-Agentic-AI-Assistant
Small agent, big brain. Processing the world one pixel at a time. 🤖
<div align="center">

# 🤖 Multi-Modal Agentic AI Assistant

### 🚀 *Production-Grade Intelligent AI System with Model Routing, Tool Calling & Ultra-Fast Inference*

<p>
<img src="https://img.shields.io/badge/Python-3.9+-blue.svg"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-AgenticAI-green"/>
<img src="https://img.shields.io/badge/HuggingFace-Inference-yellow"/>
<img src="https://img.shields.io/badge/AI-Agentic%20Systems-purple"/>
</p>

</div>

---

# 🧠 The Problem

Traditional AI systems:

* Depend on a **single large model**
* Are **slow and expensive**
* Lack **task specialization**
* Cannot scale efficiently in production environments

This leads to:

* High latency
* Poor user experience
* Inefficient resource usage

---

# 💡 The Solution

This project implements a **Multi-Modal Agentic AI Assistant** that:

* Dynamically **routes queries to specialized models**
* Uses **lightweight LLMs for ultra-fast responses**
* Leverages **Hugging Face Inference API**
* Follows a **modern Agentic AI architecture**

---

# ✨ Key Highlights

### ⚡ Ultra-Fast AI System

* Uses **small language models (1B–4B)**
* Optimized for **low-latency inference (<1–2 sec)**
* Designed for **Streamlit Cloud constraints**

---

### 🧠 Intelligent Model Routing

| Task Type          | Model                 |
| ------------------ | --------------------- |
| Sensitive Data     | SmolLM2-135M          |
| General Chat       | Phi-3 Mini            |
| Fallback           | Gemma 2B              |
| Advanced Reasoning | Mistral 7B (API only) |

---

### 🔄 Agentic AI Architecture

Unlike simple chatbots, this system:

* Understands user intent
* Plans execution
* Selects tools/models
* Generates intelligent responses

---

### 🔗 Hugging Face API Integration

* No local model loading
* Fully cloud-compatible
* Scalable and efficient

---

### 🧰 Tool-Enabled System

Supports:

* Calculator tools
* File handling
* Extendable API integrations

---

### 🧠 Memory System

* Maintains conversation context
* Enables more human-like interactions

---

# 🏗️ System Architecture

```text
User
 ↓
Streamlit UI
 ↓
Agent Runtime (LangChain / LangGraph)
 ↓
Model Router
 ↓
Hugging Face Inference API
 ↓
Selected Model
 ↓
Response
```

---

# ⚙️ Tech Stack

| Layer           | Technologies                        |
| --------------- | ----------------------------------- |
| Frontend        | Streamlit                           |
| Backend         | Python                              |
| Agent Framework | LangChain, LangGraph                |
| LLM APIs        | Hugging Face Inference API          |
| Models          | Phi-3 Mini, SmolLM2, Gemma, Mistral |
| Memory          | LangChain Memory                    |
| Deployment      | Streamlit Community Cloud           |

---

# 🔄 Workflow

```text
User Query
   ↓
Intent Detection
   ↓
Model Router
   ↓
Selected Model (HF API)
   ↓
Tool Execution (if required)
   ↓
Response Generation
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/agentic-ai-assistant.git
cd agentic-ai-assistant
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Add Environment Variables

Create `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 4️⃣ Run Application

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# ☁️ Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Deploy via Streamlit Cloud
3. Add secret:

```toml
HUGGINGFACEHUB_API_TOKEN="your_token"
```

---

# 📊 Performance Optimization

* Uses **lightweight models for speed**
* Reduces **token usage**
* Implements **fallback handling**
* Avoids heavy local computation

---

# 🔮 Future Enhancements

* Multi-agent system (Planner + Executor)
* RAG-based knowledge system
* Voice assistant integration
* Real-time web search tool
* Database integration

---

# 🧪 Real-World Use Cases

* AI Assistants
* Workflow Automation
* Data Analysis Tools
* Smart Chatbots
* Developer Assistants

---

# 👨‍💻 Author

### Abdul Azeem Sheikh

**Information Science Engineer | AI & ML Developer**

* 💡 Specialized in **Agentic AI, LLMs, and Automation Systems**
* 🧠 Experience in **LangChain, RAG, and AI Agents**
* ⚙️ Built real-world AI solutions using **Streamlit & Python**

🌐 Portfolio
https://azeemsheikh.vercel.app/

📧 Email
[abdulazeemsheik4@gmail.com](mailto:abdulazeemsheik4@gmail.com)

💼 GitHub
https://github.com/azzu-sheikh

---

# ⭐ Support & Contribution

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with the community

---

<div align="center">

### 🚀 *Building the Future with Agentic AI Systems*

</div>
