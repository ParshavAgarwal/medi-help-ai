# 🩺 MediHelp AI

**MediHelp AI** is an intelligent health companion web app built with Streamlit and Groq's model. It helps users understand their symptoms, decode medical reports and scans, and get quick, science-backed answers to health questions — all through a simple conversational interface.

> ⚠️ **Disclaimer:** MediHelp AI is an educational tool and does **not** replace professional medical advice, diagnosis, or treatment. Always consult a licensed doctor for personal health decisions. In case of a medical emergency, call your local emergency number immediately.

---

## ✨ Features

### 🩺 1. Virtual Clinic — AI Symptom Checker
- Describe your symptoms in plain language.
- Built-in **emergency keyword detection** flags potentially life-threatening symptoms (chest pain, difficulty breathing, stroke signs, etc.) instantly and advises the user to seek emergency care — before any AI call is made.
- If not an emergency, the AI asks **4 targeted follow-up questions** (onset/duration, severity, associated symptoms, aggravating/relieving factors) to gather more context — just like a real triage conversation.
- Based on your answers, generates a structured health assessment covering:
  - Likely Condition
  - Severity Level
  - Home Remedies
  - Over-the-Counter Medicines
  - Whether You Should See a Doctor
  - Warning Signs to Watch For
  - A Lifestyle Tip
- Download the full assessment as a formatted PDF.

### 📋 2. Medical Report Analyzer
- Upload a medical image (X-Ray, MRI, CT Scan, Ultrasound) **or** paste lab report text (CBC, Lipid Panel, LFT, KFT, HbA1c, Thyroid Profile, etc.).
- Uses Gemini's multimodal (vision + text) capability to analyze the report/image and translate medical jargon into plain English.
- Structured output includes:
  - Summary of Results
  - Abnormal Values
  - Normal Values
  - What This Means for You
  - Recommended Actions
  - A short Plain English Summary
- Export the analysis as a PDF or plain text file.

### 💬 3. Medical Knowledge Chatbot
- Ask open-ended questions about health, anatomy, medications, lab values, and nutrition.
- Comes with quick-start question buttons for common queries.
- Maintains recent conversation context for more natural follow-up questions.
- Stays strictly within health/medical topics and always recommends professional consultation for personal decisions.
- Export the full chat session as PDF or text.

### 👤 Patient Profile Personalization
- A sidebar profile (name, age, gender, location, known conditions, allergies) is used to personalize every AI response across all three features — without any model fine-tuning, purely through prompt context injection.

### 📄 Branded PDF Export
- Every feature supports exporting results as a clean, branded PDF report with a custom header/footer, patient info box, and formatted sections — powered by a custom `fpdf2` subclass.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend & App Framework | [Streamlit](https://streamlit.io/) |
| AI Model | [Groq qwen/qwen3.8-27b — text + vision |
| Image Handling | Pillow (PIL) |
| PDF Generation | fpdf2 |
| Environment Config | python-dotenv |
| Language | Python 3.9+ |

---

## 📂 Project Structure

```
medi-help-ai/
├── app2.py            # Main Streamlit application (all UI + logic)
├── requirements.txt    # Python dependencies
└── .env                # Your Gemini API key (not committed to git)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd medi-help-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free Groq API key from [Groq Console](https://console.groq.com/keys).

### 5. Run the app
```bash
streamlit run app2.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 How It Works (High-Level Flow)

1. User fills in an optional patient profile in the sidebar (name, age, gender, conditions, allergies).
2. Based on the selected tab (Virtual Clinic / Report Analyzer / Chatbot), user input (text or image) is sent to Gemini along with a carefully structured prompt that includes patient context.
3. Gemini returns a structured, plain-English response using fixed section headers.
4. The response is rendered in the UI and can be exported as a PDF (via a custom `fpdf2` layout) or plain text.
5. Session data (chat history, symptom-check stage, profile) is held in Streamlit's `session_state` for the duration of the browser session.

---

## 🚧 Known Limitations

- No persistent database — all data is session-based and lost on refresh/tab close.
- No user authentication or multi-user accounts.
- Emergency detection is currently keyword-based and may not catch all phrasings of urgent symptoms.
- No automated evaluation pipeline for AI response accuracy.

---

## 🔮 Future Scope

- Add persistent storage (PostgreSQL) for user accounts and health history.
- Replace keyword-based emergency detection with a semantic/NLP-based classifier.
- Add a secondary AI "safety reviewer" pass to validate medical responses before showing them.
- Multi-language support (Hindi and other regional languages).
- Rate limiting and cost controls for API usage.

---

## 📜 License

This project is for educational/demonstration purposes.
