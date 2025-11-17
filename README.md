---

## 🚀 **Features**

### 🔍 **1. Ask Questions (RAG-based Q&A)**

* Upload PDF/TXT files
* Automatically chunk and embed content
* Ask natural-language questions
* AI answers strictly based on document context

### 📝 **2. Notes Generation**

* Generates structured, well-organized notes
* Headings, bullets, and logical flow
* Export as `.txt`

### 📋 **3. Short Notes / Summaries**

* 5–10 bullet-point summaries
* Useful for quick revision
* Export supported

### ❓ **4. Quiz Generation**

* Auto-generates MCQs from the document
* Includes answer keys
* Adjustable question count

### ⚡ **5. Fast & Lightweight Architecture**

* Qdrant **in-memory** vector store (no setup required)
* HuggingFace MiniLM embeddings (fast on CPU)
* Groq API for lightning-fast inference
* Streamlit frontend for instant deployment

---

## 🧩 **Tech Stack**

| Layer            | Technology                            |
| ---------------- | ------------------------------------- |
| **Frontend**     | Streamlit                             |
| **Embeddings**   | SentenceTransformers MiniLM-L6-v2     |
| **Vector Store** | Qdrant (in-memory)                    |
| **LLM**          | Groq — Llama-3.1-8B-Instant           |
| **Backend**      | Python                                |
| **File Parsing** | PyPDF, TextLoader                     |
| **RAG Pipeline** | Custom (no LangChain vector wrappers) |

---

## 📦 **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/prep-pal.git
cd prep-pal
```

### **2. Create a virtual environment**

```bash
python -m venv env
env\Scripts\activate     # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set your Groq API key**

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

---

## ▶️ **Run the App**

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 🗂 **Project Structure**

```
prep-pal/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # All dependencies
├── README.md             # Project documentation
│
└── (Qdrant runs in memory; no DB files created)
```

---

## 📘 **How It Works**

1. **Upload File**
   PDF or TXT is parsed into raw text.

2. **Text Splitting**
   RecursiveCharacterTextSplitter chunks the document.

3. **Embedding**
   MiniLM model converts chunks into embeddings.

4. **Vector Storage**
   In-memory Qdrant stores embeddings for fast retrieval.

5. **Query Time**
   User query → embedded → similarity search in Qdrant.

6. **Context Passed to Groq LLM**
   Groq generates:

   * Q&A answers
   * Notes
   * Summaries
   * Quizzes

---

## ⭐ **Why Prep Pal?**

* Designed to be **simple**, **fast**, and **lightweight**
* No database setup, no heavy GPUs, no complicated framework
* Ideal for students, developers, or anyone who needs smart study tools
* Clean architecture that's easy to extend

---

## 🤝 **Contributing**

Contributions are welcome!
Fork the repo, open an issue, or submit a pull request.

---

## 📜 **License**

This project is licensed under the MIT License.

---

## 💬 **Feedback**

If you have suggestions or feature requests:
👉 Open an issue on GitHub or DM the owner.

---
