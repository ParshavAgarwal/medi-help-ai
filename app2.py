import streamlit as st
import google.generativeai as genai
from PIL import Image
from duckduckgo_search import DDGS
from fpdf import FPDF
from dotenv import load_dotenv
import os
import io
import re
import datetime
import unicodedata

# ── Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# ── Page config
st.set_page_config(
    page_title="MediHelp AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# basic CSS
st.markdown(
    """
<style>
/* Emergency pulsing banner — no native Streamlit equivalent */
.emergency-box {
    background-color: #7f1d1d;
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    animation: pulse 1.2s ease-in-out infinite;
    margin: 0.5rem 0;
}
.emergency-box h3 { color: #fca5a5; margin: 0 0 0.3rem; }
.emergency-box p  { color: #fecaca; margin: 0; font-size: 0.95rem; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%       { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
}
/* Chat bubbles — layout not possible with pure Streamlit */
.bubble-user {
    background: #1e293b;
    border-radius: 12px 12px 4px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem 20%;
    font-size: 0.93rem;
    color: #e2e8f0;
}
.bubble-ai {
    background: #0f2027;
    border: 1px solid #0d9488;
    border-radius: 12px 12px 12px 4px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 20% 0.4rem 0;
    font-size: 0.93rem;
    color: #e2e8f0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Default Session state
_defaults = {
    "patient_profile": {
        "name": "",
        "age": "",
        "gender": "Male",
        "location": "Jaipur, Rajasthan",
        "conditions": "",
        "allergies": "",
    },
    "symptom_chat": [],
    "symptom_stage": "init",
    "initial_symptoms": "",
    "verdict_text": "",
    "report_analysis": "",
    "chat_history": [],
    "_ref_results": [],
    "_ref_term": "",
    "_ref_fetched": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper: Gemini text
def gemini_text(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"❌ Error: {e}"


# ── Helper: Gemini image (PIL path — avoids SDK hang)
def gemini_image(prompt: str, image_bytes: bytes) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return model.generate_content([img, prompt]).text
    except Exception as e:
        return f"❌ Error: {e}"


# ── Helper: DuckDuckGo search with retry
def ddg_search(query: str, max_results: int = 6) -> list:
    for _ in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results
        except Exception:
            pass
    return []


# ── Helper: Medical references — trusted domains, multiple query fallback ──────
def fetch_references(condition: str) -> list:
    trusted_domains = [
        "mayoclinic",
        "healthline",
        "webmd",
        "medlineplus",
        "nhs.uk",
        "nih.gov",
        "clevelandclinic",
        "hopkinsmedicine",
        "medicalnewstoday",
        "who.int",
    ]
    queries = [
        f"{condition} causes symptoms treatment",
        f"{condition} medical information diagnosis",
        f"what is {condition} health guide",
    ]
    for q in queries:
        results = ddg_search(q, max_results=8)
        trusted = [
            r
            for r in results
            if r.get("href") and any(d in r["href"] for d in trusted_domains)
        ]
        if trusted:
            return trusted[:5]
    # last resort — return raw results from broadest query
    return ddg_search(f"{condition} health", max_results=5)


# ── Helper: Emergency keyword check
def is_emergency(text: str) -> bool:
    flags = [
        "chest pain",
        "heart attack",
        "can't breathe",
        "cannot breathe",
        "difficulty breathing",
        "stroke",
        "unconscious",
        "fainted",
        "seizure",
        "severe bleeding",
        "suicide",
        "overdose",
        "anaphylaxis",
        "not breathing",
        "coughing blood",
        "vomiting blood",
        "paralysis",
        "loss of vision",
        "sudden blindness",
    ]
    return any(f in text.lower() for f in flags)


# ── PDF helpers
def sanitize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    safe = "".join(c for c in nfkd if ord(c) < 256)
    for orig, repl in {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2022": "*",
        "\u00b0": " deg",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
    }.items():
        safe = safe.replace(orig, repl)
    return safe.encode("latin-1", errors="replace").decode("latin-1")


def strip_md(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text


class MediPDF(FPDF):
    _rtype = "Report"

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(15, 118, 110)
        self.rect(0, 0, 210, 10, "F")
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(255, 255, 255)
        self.set_xy(15, 2)
        self.cell(90, 6, sanitize(f"MediHelp AI  |  {self._rtype}"), ln=False)
        self.set_xy(105, 2)
        self.cell(90, 6, f"Page {self.page_no()}", ln=False, align="R")
        self.set_y(13)

    def footer(self):
        self.set_y(-11)
        self.set_draw_color(180, 200, 195)
        self.set_line_width(0.2)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", "I", 6.5)
        self.set_text_color(150, 150, 150)
        self.cell(0, 4, "If Confused, also consult with a Human Doctor", align="C")


def _write(pdf: FPDF, text: str, h: float = 5.0):
    text = sanitize(text).strip()
    if not text:
        return
    w = pdf.w - pdf.l_margin - pdf.r_margin
    for i in range(0, len(text), 120):
        chunk = text[i : i + 120].strip()
        if chunk:
            try:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w, h, chunk)
            except Exception:
                pass


def make_pdf(
    title: str,
    content: str,
    profile: dict,
    report_type: str = "Report",
    references: list = None,
) -> bytes:
    pdf = MediPDF(orientation="P", unit="mm", format="A4")
    pdf._rtype = report_type
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(15, 118, 110)
    pdf.rect(0, 0, 210, 24, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(0, 4)
    pdf.cell(0, 9, "MediHelp AI", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, sanitize(report_type), ln=True, align="C")
    pdf.ln(4)

    # Timestamp
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.set_x(15)
    pdf.cell(
        0,
        5,
        f"Generated: {datetime.datetime.now().strftime('%d %b %Y  |  %I:%M %p')}",
        ln=True,
        align="R",
    )
    pdf.ln(2)

    # Patient box
    p = profile
    name = sanitize(p.get("name", "") or "N/A")
    age = sanitize(str(p.get("age", "")) or "N/A")
    gen = sanitize(p.get("gender", "N/A"))
    loc = sanitize(p.get("location", "N/A"))
    cond = sanitize(p.get("conditions", "") or "None")
    allg = sanitize(p.get("allergies", "") or "None")
    y0 = pdf.get_y()
    pdf.set_fill_color(236, 253, 245)
    pdf.rect(15, y0, 180, 20, "F")
    pdf.set_xy(18, y0 + 2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(5, 80, 60)
    pdf.cell(0, 4, "PATIENT PROFILE", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(18)
    pdf.cell(
        0,
        4.5,
        f"Name: {name}    Age: {age}    Gender: {gen}    Location: {loc}",
        ln=True,
    )
    pdf.set_x(18)
    pdf.cell(0, 4.5, f"Conditions: {cond}    Allergies: {allg}", ln=True)
    pdf.ln(6)

    # Title
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(5, 100, 80)
    pdf.set_x(15)
    pdf.multi_cell(180, 7, sanitize(title))
    pdf.set_draw_color(15, 118, 110)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Content
    orig_lines = content.split("\n")
    stripped_lines = strip_md(content).split("\n")
    while len(stripped_lines) < len(orig_lines):
        stripped_lines.append("")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(35, 35, 35)

    for orig, stripped in zip(orig_lines, stripped_lines):
        clean = sanitize(stripped).strip()
        if not clean:
            pdf.ln(2)
            continue

        if orig.strip().startswith("#"):  # section header
            pdf.ln(3)
            pdf.set_fill_color(236, 253, 245)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(5, 100, 80)
            pdf.set_x(15)
            pdf.multi_cell(180, 6, clean, fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(35, 35, 35)
            pdf.ln(1)

        elif orig.strip().startswith(("-", "*", "•")):  # bullet
            btext = sanitize(re.sub(r"^[-*•]+\s*", "", orig)).strip()
            if btext:
                y_now = pdf.get_y()
                pdf.set_xy(20, y_now)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(5, 5, "-", ln=False)
                pdf.set_x(25)
                for i in range(0, len(btext), 115):
                    chunk = btext[i : i + 115].strip()
                    if chunk:
                        try:
                            pdf.multi_cell(165, 5, chunk)
                            pdf.set_x(25)
                        except Exception:
                            pass
        else:  # paragraph
            _write(pdf, clean, 5.0)

    # References section
    if references:
        pdf.ln(5)
        pdf.set_fill_color(219, 234, 254)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.set_x(15)
        pdf.multi_cell(180, 6, "Trusted Medical References", fill=True)
        pdf.ln(2)
        for i, ref in enumerate(references[:6], 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(35, 35, 35)
            _write(pdf, f"{i}. {ref.get('title', '')}", 5.0)
            if ref.get("href"):
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(30, 64, 175)
                _write(pdf, ref["href"], 4.5)
            if ref.get("body"):
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(80, 80, 80)
                _write(pdf, ref["body"][:200], 4.5)
            pdf.set_text_color(35, 35, 35)
            pdf.ln(2)

    return bytes(pdf.output())


# SIDEBAR

with st.sidebar:
    st.title("🩺 MediHelp AI")
    st.caption("Intelligent Health Companion")
    st.divider()

    st.subheader("👤 Patient Profile")
    p = st.session_state.patient_profile
    p["name"] = st.text_input(
        "Full Name", value=p["name"], placeholder="e.g. Rohan Sharma"
    )
    p["age"] = st.text_input("Age", value=p["age"], placeholder="e.g. 28")
    p["gender"] = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(p.get("gender", "Male")),
    )
    p["location"] = st.text_input(
        "Location", value=p["location"], placeholder="City, State"
    )
    p["conditions"] = st.text_input(
        "Known Conditions", value=p.get("conditions", ""), placeholder="e.g. Diabetes"
    )
    p["allergies"] = st.text_input(
        "Allergies", value=p.get("allergies", ""), placeholder="e.g. Penicillin"
    )
    st.session_state.patient_profile = p


# MAIN HEADER

p = st.session_state.patient_profile
greeting = p["name"].split()[0] if p["name"] else "there"

st.title(f"🩺 Hello, {greeting}!")
subtitle_parts = ["How can MediHelp AI help you today?"]
if p["location"]:
    subtitle_parts.append(f"📍 {p['location']}")
if p["age"]:
    subtitle_parts.append(f"🎂 Age {p['age']}")
st.caption("  •  ".join(subtitle_parts))
st.info("⚡ Available 24/7 — Powered by Gemini 2.5 Flash")
st.divider()


tab1, tab2, tab3 = st.tabs(
    ["🩺 Virtual Clinic", "📋 Report Analyzer", "💬 Medical Chatbot"]
)


# TAB 1 — VIRTUAL CLINIC

with tab1:
    left_col, right_col = st.columns([2.2, 1], gap="large")

    with right_col:
        st.subheader("🚨 Emergency Numbers")
        st.error(
            "**National Emergency: 112**\n\nAmbulance: 108\n\nAIIMS Helpline: 1800-11-2444\n\nCardiac: 1800-112-132\n\nMental Health (iCall): 9152987821"
        )

        st.subheader("📊 Severity Guide")
        st.success("🟢 **Mild** — Home remedies work")
        st.warning("🟡 **Moderate** — See doctor in 24-48h")
        st.error("🔴 **Severe** — Seek immediate care")

    with left_col:
        st.subheader("Symptom Checker")
        symptoms_input = st.text_area(
            "Describe your symptoms",
            placeholder="e.g. Sharp headache for 2 hours, feeling dizzy and nauseous...",
            height=110,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            go = st.button(
                "🔍 Start Analysis", use_container_width=True, type="primary"
            )
        with btn_col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.symptom_chat = []
                st.session_state.symptom_stage = "init"
                st.session_state.initial_symptoms = ""
                st.session_state.verdict_text = ""
                st.rerun()

        # Emergency check — always first
        if symptoms_input and is_emergency(symptoms_input):
            st.markdown(
                """
            <div class="emergency-box">
                <h3>🚨 CALL 112 IMMEDIATELY 🚨</h3>
                <p>Your symptoms may indicate a life-threatening emergency.<br>
                <strong>Do NOT wait for AI — call 112 or go to the ER right now.</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Stage 1 — ask follow-up questions
        if go and symptoms_input and not is_emergency(symptoms_input):
            st.session_state.initial_symptoms = symptoms_input
            st.session_state.symptom_chat = [
                {"role": "user", "content": symptoms_input}
            ]
            st.session_state.symptom_stage = "followup"
            with st.spinner("Consulting AI doctor..."):
                resp = gemini_text(f"""You are a compassionate expert medical AI.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}, Location {p["location"]}.
Conditions: {p["conditions"] or "none"}. Allergies: {p["allergies"] or "none"}.
Symptoms: "{symptoms_input}"

1. Acknowledge warmly (1-2 sentences).
2. Ask exactly 4 numbered follow-up questions to clarify onset/duration, severity (1-10), associated symptoms, and aggravating/relieving factors.
Do NOT diagnose yet.""")
            st.session_state.symptom_chat.append({"role": "ai", "content": resp})
            st.rerun()

        # Display chat
        if st.session_state.symptom_chat:
            st.divider()
            st.subheader("Consultation")
            for msg in st.session_state.symptom_chat:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="bubble-user"><strong>You:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="bubble-ai"><strong>🤖 MediHelp AI:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Stage 2 — collect answers, generate verdict
        if st.session_state.symptom_stage == "followup":
            answers = st.text_area(
                "Your answers to the follow-up questions:",
                placeholder="1. Started yesterday...\n2. Severity is 6/10...",
                height=120,
            )
            if st.button(
                "✅ Get Full Assessment", use_container_width=True, type="primary"
            ):
                if answers.strip():
                    st.session_state.symptom_chat.append(
                        {"role": "user", "content": answers}
                    )
                    with st.spinner(
                        "Generating your personalized health assessment..."
                    ):
                        verdict = gemini_text(f"""You are a senior medical AI giving a thorough clinical assessment.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}, Location {p["location"]}.
Conditions: {p["conditions"] or "none"}. Allergies: {p["allergies"] or "none"}.
Symptoms: "{st.session_state.initial_symptoms}"
Follow-up answers: "{answers}"

Use EXACTLY these headers:
## Likely Condition
## Severity Level
## Home Remedies
## Over-the-Counter Medicines
## Should You See a Doctor?
## Warning Signs to Watch For
## Lifestyle Tip

Be warm, clear, evidence-based.""")
                    st.session_state.symptom_chat.append(
                        {"role": "ai", "content": verdict}
                    )
                    st.session_state.verdict_text = verdict
                    st.session_state.symptom_stage = "verdict"
                    st.rerun()

        # Stage 3 — PDF download
        if (
            st.session_state.symptom_stage == "verdict"
            and st.session_state.verdict_text
        ):
            st.divider()
            pdf_bytes = make_pdf(
                "Virtual Clinic Assessment",
                st.session_state.verdict_text,
                p,
                "Virtual Clinic Report",
            )
            st.download_button(
                "📄 Download Assessment as PDF",
                data=pdf_bytes,
                file_name="medihelp_clinic.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# TAB 2 — REPORT ANALYZER

with tab2:
    st.subheader("Medical Report Analyzer")
    st.caption(
        "Upload a scan or paste lab results — AI translates jargon into plain English. References + PDF included."
    )

    report_mode = st.radio(
        "Report type:",
        [
            "🖼️ Medical Image (X-Ray / MRI / CT / Ultrasound)",
            "🧪 Lab Report / Blood Test",
            "💊 Prescription / Doctor Notes",
        ],
        horizontal=True,
    )

    # ── Input section
    if "Image" in report_mode:
        img_col, prev_col = st.columns([1, 1])
        with img_col:
            uploaded = st.file_uploader(
                "Upload medical image", type=["jpg", "jpeg", "png"]
            )
        with prev_col:
            if uploaded:
                st.session_state["_img_bytes"] = uploaded.getvalue()
                st.image(
                    Image.open(io.BytesIO(st.session_state["_img_bytes"])),
                    caption=uploaded.name,
                    use_container_width=True,
                )

        if uploaded:
            if st.button("🔬 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner(
                    "🧠 AI is reading your scan — usually 15-30 seconds..."
                ):
                    result = gemini_image(
                        f"""You are a consultant radiologist.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}.
Conditions: {p["conditions"] or "none"}.

Analyze thoroughly. Use EXACTLY these headers:
## Image Type and Region
## Technical Quality
## Key Findings
## Diagnostic Assessment
## Plain English Explanation
## Recommended Next Steps
## Important Note (confirm AI must be reviewed by licensed radiologist)""",
                        st.session_state.get("_img_bytes", uploaded.getvalue()),
                    )

                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.session_state.report_analysis = result
                    st.session_state["_ref_results"] = []
                    st.session_state["_ref_fetched"] = False
                    st.session_state["_ref_term"] = ""
                    st.rerun()
    else:
        report_text = st.text_area(
            "Paste report content:",
            height=200,
            placeholder="e.g.\nHbA1c: 8.2%\nFasting Glucose: 178 mg/dL\nLDL: 145 mg/dL\n...",
        )
        if st.button(
            "🔬 Analyze Report",
            use_container_width=True,
            key="analyze_text_btn",
            type="primary",
        ):
            if report_text.strip():
                with st.spinner("Analyzing your report..."):
                    result = gemini_text(f"""You are a senior clinical pathologist.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}.
Conditions: {p["conditions"] or "none"}. Allergies: {p["allergies"] or "none"}.
Report type: {report_mode}
Report:
{report_text}

Use EXACTLY these headers:
## Summary of Results
## Abnormal Values
## Normal Values
## What This Means for You
## Recommended Actions
## Plain English Summary (3 sentences max)""")

                st.session_state.report_analysis = result
                st.session_state["_ref_results"] = []
                st.session_state["_ref_fetched"] = False
                st.session_state["_ref_term"] = ""
                st.rerun()

    # ── Results — full width below
    if st.session_state.report_analysis:
        st.divider()
        st.subheader("📄 Analysis Results")
        st.markdown(st.session_state.report_analysis)
        st.divider()

        # Auto-fetch references once
        if not st.session_state.get("_ref_fetched"):
            term_raw = gemini_text(
                "Extract ONLY the most important medical condition/finding from this text. "
                "Reply with 2-4 words only:\n" + st.session_state.report_analysis[:500]
            )
            term = term_raw.strip().split("\n")[0].strip()
            st.session_state["_ref_term"] = term
            st.session_state["_ref_fetched"] = True

        # Show references
        refs = st.session_state.get("_ref_results", [])
        term = st.session_state.get("_ref_term", "")

        st.divider()

        # Download buttons
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            pdf_bytes = make_pdf(
                "Medical Report Analysis",
                st.session_state.report_analysis,
                p,
                "Report Analysis",
                references=refs or None,
            )
            st.download_button(
                "📥 Download PDF (with references)",
                data=pdf_bytes,
                file_name="medihelp_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with dl2:
            txt = st.session_state.report_analysis
            if refs:
                txt += "\n\n--- References ---\n"
                for r in refs:
                    txt += f"\n• {r.get('title', '')}\n  {r.get('href', '')}\n  {r.get('body', '')[:200]}\n"
            st.download_button(
                "📝 Download as Text",
                data=txt,
                file_name="medihelp_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl3:
            if st.button("🗑️ Clear & New Analysis", use_container_width=True):
                st.session_state.report_analysis = ""
                st.session_state["_ref_results"] = []
                st.session_state["_ref_term"] = ""
                st.session_state["_ref_fetched"] = False
                st.session_state.pop("_img_bytes", None)
                st.rerun()
    else:
        st.info("Upload a medical image or paste your lab report above to get started.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📖 How it works")
            st.markdown("""
1. Choose your report type
2. Upload an image **or** paste text
3. AI translates medical jargon into plain English
4. References auto-fetched from Mayo Clinic, Healthline, WebMD
5. Download full PDF (with references) or plain text
""")
        with c2:
            st.subheader("✅ Supported types")
            st.markdown("""
- X-Ray, MRI, CT Scan, Ultrasound images
- CBC, Lipid Panel, LFT, KFT, HbA1c
- Thyroid Profile, Vitamin & Hormone panels
- Doctor prescriptions and clinical notes
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MEDICAL CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Medical Knowledge Chatbot")
    st.caption(
        "Ask anything about health, anatomy, medications, lab values, nutrition — science-backed answers."
    )

    # Quick starters
    st.markdown("**⚡ Quick questions:**")
    qs_cols = st.columns(3)
    starters = [
        ("💉", "Why does my arm hurt after a vaccine?"),
        ("🩸", "What does a high WBC count mean?"),
        ("😴", "Why do I feel sleepy after meals?"),
        ("❤️", "How does stress affect heart health?"),
        ("🥗", "Best anti-inflammatory foods?"),
        ("💊", "Ibuprofen vs Paracetamol — when to use which?"),
    ]
    for i, (icon, q) in enumerate(starters):
        with qs_cols[i % 3]:
            if st.button(f"{icon} {q}", key=f"qs_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    ans = gemini_text(f"""You are a friendly expert medical educator.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}.
Question: {q}
Answer clearly (150-250 words). Define terms in brackets. Add 1 practical tip.
End with: 'For personal decisions, consult your doctor.'""")
                st.session_state.chat_history.append({"role": "ai", "content": ans})
                st.rerun()

    st.divider()

    # Chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-16:]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="bubble-user"><strong>You:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bubble-ai"><strong>🤖 MediHelp AI:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "💬 Ask me anything about health science, medications, lab values, symptoms, nutrition, and more."
        )

    # Input
    inp_c, btn_c = st.columns([5, 1])
    with inp_c:
        user_q = st.text_input(
            "Your question...",
            key="chat_q",
            label_visibility="collapsed",
            placeholder="e.g. How does insulin regulate blood sugar?",
        )
    with btn_c:
        send = st.button("Send ➤", use_container_width=True, type="primary")

    if send and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        ctx = "\n".join(
            [
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][-300:]}"
                for m in st.session_state.chat_history[-6:]
            ]
        )
        with st.spinner("Thinking..."):
            reply = gemini_text(f"""You are a friendly expert medical educator.
Only discuss: health, medicine, anatomy, nutrition, medications, lab values.
Patient: Age {p["age"] or "unknown"}, Gender {p["gender"]}, Location {p["location"]}.
Conditions: {p["conditions"] or "none"}.

Recent conversation:
{ctx}

Question: {user_q}

Instructions:
- Answer clearly and accurately. Define medical terms in brackets.
- For personal symptoms, answer educationally and recommend professional consultation.
- If off-topic from health/medicine, politely redirect.
- Max 300 words.""")
        st.session_state.chat_history.append({"role": "ai", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with ac2:
            chat_txt = "\n\n".join(
                [
                    f"{'You' if m['role'] == 'user' else 'MediHelp AI'}: {m['content']}"
                    for m in st.session_state.chat_history
                ]
            )
            pdf_bytes = make_pdf("Medical Q&A Session", chat_txt, p, "Chat Export")
            st.download_button(
                "📄 Export as PDF",
                data=pdf_bytes,
                file_name="medihelp_chat.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with ac3:
            st.download_button(
                "📝 Export as TXT",
                data=chat_txt if st.session_state.chat_history else "",
                file_name="medihelp_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )
