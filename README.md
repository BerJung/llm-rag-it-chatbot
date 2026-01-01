
---

# **📌 IT Support RAG Chatbot — FastAPI + ChromaDB + Slack**

> **Retrieval-Augmented LLM Chatbot for Internal IT Support**
> Built with **FastAPI, ChromaDB, Slack Slash Command, OpenAI API, Docker**

This project implements an **internal IT Support knowledge chatbot** that
automates repetitive helpdesk inquiries — such as password resets, VPN issues, and device requests —
using a **Retrieval-Augmented Generation (RAG)** architecture and integrates directly with Slack via `/rag`.

---

## 🚀 **Key Features**

| Capability                         | Description                                                |
| ---------------------------------- | ---------------------------------------------------------- |
| **RAG-based IT knowledge search**  | Answers grounded in internal documents (`data/it/*.txt`)   |
| **Slack Slash Command support**    | `/rag <question>` returns an immediate answer inside Slack |
| **FastAPI REST API**               | Standard `/chat` endpoint for programmatic usage           |
| **Vector database using ChromaDB** | Efficient retrieval with embeddings                        |
| **OpenAI GPT-4o-mini integration** | Enhanced contextual response quality                       |
| **Production-oriented structure**  | Logging, environment variables, configurable models        |
| **Dockerized deployment**          | Consistent runtime across environments                     |

---

## 🏢 **Use Case Overview**

This chatbot **reduces internal support load** by automatically responding to frequent IT questions:

* “How do I reset my password?”
* “VPN won’t connect — what should I check?”
* “How can I request a new laptop?”
* “What’s the guest Wi-Fi process?”

> Designed as a practical foundation for future team-specific bots
> (HR / Operations / Engineering documentation can be added later).

---

## 🧱 **Tech Stack**

* **Python 3.11**
* **FastAPI**
* **ChromaDB** (vector store)
* **OpenAI API** (GPT-4o-mini, embeddings)
* **Slack API** (Slash Command integration)
* **Docker**
* **Ubuntu / WSL2**

---

## 📂 **Project Structure**

```
llm-bot/
├── app/
│   ├── config.py        # environment & settings
│   ├── main.py          # FastAPI app, RAG-enabled endpoints
│   └── rag.py           # document loading + embeddings + retrieval
├── data/
│   └── it/              # IT support documents used as RAG sources
│       ├── device_request_policy.txt
│       ├── password_reset.txt
│       ├── vpn_troubleshooting.txt
│       └── wifi_access.txt
├── chroma_db/           # persistent vector store
├── .env                 # API keys + configuration (not committed)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ **Environment Setup**

### 1️⃣ Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2️⃣ Create `.env`

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=4
LOG_LEVEL=INFO
```

> **Do not commit `.env` to Git.**

---

## 🔄 **Run the Vector Store Build + API Server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server automatically builds vector store on startup.

---

## 📡 **API Usage**

### `POST /chat`

```json
{
  "message": "VPN keeps disconnecting. What should I check?",
  "user_id": "example_user",
  "channel": "web",
  "use_rag": true
}
```

**Response example:**

```json
{
  "reply": "First verify your network connection...",
  "timestamp": "2025-02-10T21:33:12Z",
  "model": "gpt-4o-mini",
  "used_rag": true
}
```

---

## 💬 **Slack Integration**

1️⃣ Start server normally
2️⃣ Run ngrok (development)

```bash
ngrok http 8000
```

3️⃣ Copy `https://xxxxx.ngrok.app/slack/slash`
4️⃣ Add to Slack App → **Slash Command `/rag`**

**Example (Slack):**

```
/rag How do I reset my password?
/rag VPN won’t connect — what should I check?
/rag How can I request a new laptop?
```

---

## 📝 **Logging Example**

```
2025-02-10 21:22:41 [INFO] llm_bot -
chat 요청 처리: user_id=nathan channel=slack used_rag=True len_q=42 duration=2.13s
```

---

## 📌 **Next Steps / Roadmap**

* Add **HR & Ops scenario docs** for multi-domain support
* Deploy to **Render / Railway / Fly.io**
* Add **Teams integration**
* Store **chat history** to track repeated questions
* Add **observability dashboard** (Prometheus/Grafana)

---

## 💡 **Recruiter-friendly Summary**

> **Built an internal RAG chatbot with Slack integration to automate IT Support workflows.
> Uses ChromaDB vector search and GPT-4o-mini to return contextual answers on password reset, VPN issues, and device requests, reducing repetitive tickets and improving response time.**

---

## 📄 **License**

MIT License

-
