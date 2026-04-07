import streamlit as st
from groq import Groq, RateLimitError
import firebase_admin
from firebase_admin import credentials, firestore, auth
import datetime
import json
import smtplib
from email.mime.text import MIMEText
import requests
import re
import hashlib
import time
import threading
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Delivix", page_icon="💊", layout="wide")
with open("delivix_game_theme.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Firebase init (only once) ─────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["firebase"]["client_email"],
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Pinecone + Embeddings init ────────────────────────────────
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index(st.secrets["pinecone"]["index_name"])
    return index

@st.cache_resource
def init_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

pinecone_index = init_pinecone()
embedder = init_embedder()

# ── Thread-safe embedding ─────────────────────────────────────
_embed_lock = threading.Lock()

def safe_encode(text):
    with _embed_lock:
        return embedder.encode(text).tolist()

# ── Topics ────────────────────────────────────────────────────
TOPICS = {
    "Part 1": {
        "id": "part1",
        "description": "BE210 Drug Delivery - Part 1",
        "drive_id": "1kEst98JUBpjCq6_pPQW_FaIB3l1eMtpM"
    },
    "Part 2": {
        "id": "part2",
        "description": "BE210 Drug Delivery - Part 2",
        "drive_id": "1lSfTS46W5UAEr5A9ql-p4tKr2_S8Sicz"
    },
    "Part 3": {
        "id": "part3",
        "description": "BE210 Drug Delivery - Part 3",
        "drive_id": "1KGGaK_XGaDHGMuAY70549sWKjHega2oc"
    },
    "Part 4": {
        "id": "part4",
        "description": "BE210 Drug Delivery - Part 4",
        "drive_id": "1WF_xB_2WX7lWo7QoAutiRtT6ZEelwec-"
    },
    "Part 5": {
        "id": "part5",
        "description": "BE210 Drug Delivery - Part 5",
        "drive_id": "1nmadWSoKj0E9P-Gntv8x98rmF2ZD3p5Y"
    },
    "Part 6": {
        "id": "part6",
        "description": "BE210 Drug Delivery - Part 6",
        "drive_id": "16H4T935MTyJa6gVDe4oz6eVNxTBWyK_6"
    },
    "Part 7": {
        "id": "part7",
        "description": "BE210 Drug Delivery - Part 7",
        "drive_id": "10VQBabzj6-ExYzr9jaYGr4cbNefK7V9o"
    },
    "Part 8": {
        "id": "part8",
        "description": "BE210 Drug Delivery - Part 8",
        "drive_id": "13AjZd1Pb_qz2E5Tl8koKNLThIZvHPIdr"
    },
    "Part 9": {
        "id": "part9",
        "description": "BE210 Drug Delivery - Part 9",
        "drive_id": "1aJd1ce9GkzzkHJPTc2beepLUQPqyKm73"
    },
    "Part 10": {
        "id": "part10",
        "description": "BE210 Drug Delivery - Part 10",
        "drive_id": "15FGYpeiB65Y9XbJ_yCdngWt7nCnBuIxk"
    },
}

REVIEW_INTERVALS = [1, 3, 7, 14, 30, 90]
IDLE_THRESHOLD_MINS = 5

# ── System prompts ────────────────────────────────────────────
COACH_PROMPT = """You are Delivix, an expert AI mentor in drug delivery system design from IISc's BE210 course.
You coach students through clinical drug delivery challenges using Socratic questioning.

Rules:
1. Ask ONE probing question at a time — never lecture immediately
2. Guide through: biodistribution → payload → immunogenicity → manufacturing → release kinetics
3. Recommend vehicles: LNPs, AAV, Lentiviral, PLGA, ADCs, Exosomes, Hydrogels, Microneedles
4. Reference BE210 course content and real cases: Onpattro, Comirnaty, Luxturna, Glybera
5. Be like a brilliant PhD advisor — rigorous but encouraging"""

LEARN_PROMPT = """You are Delivix, teaching assistant for IISc's BE210 Drug Delivery course.
Answer student questions clearly using the lecture material provided.
Use examples, analogies, and connect concepts. Be encouraging and thorough.
Always relate answers back to the BE210 course content.
If the lecture material does not cover the question, say so clearly and suggest the student ask their instructor."""

QUIZ_PROMPT = """You are Delivix, creating a quiz for a drug delivery student.
Generate exactly {count} questions of mixed types based on the course material provided.
Do NOT use outside knowledge.
Include these question types:
- MCQ (multiple choice with 4 options)
- True/False
- Fill in the blank (use _____ for the blank)
- Match the following (provide 4 pairs)

IMPORTANT for fill_blank questions: include an "acceptable" array with synonyms and abbreviations.

Format as JSON only — no markdown, no backticks, no preamble, just the raw JSON object:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct": "A",
      "explanation": "..."
    }},
    {{
      "type": "true_false",
      "question": "...",
      "correct": "True",
      "explanation": "..."
    }},
    {{
      "type": "fill_blank",
      "question": "_____ is the process by which...",
      "correct": "exact answer word",
      "acceptable": ["synonym1", "abbreviation", "alternate phrasing"],
      "explanation": "..."
    }},
    {{
      "type": "match",
      "question": "Match the following:",
      "left": ["Term 1", "Term 2", "Term 3", "Term 4"],
      "right": ["Def A", "Def B", "Def C", "Def D"],
      "correct_pairs": {{"Term 1": "Def B", "Term 2": "Def A", "Term 3": "Def D", "Term 4": "Def C"}},
      "explanation": "..."
    }}
  ]
}}"""

# ── RAG Helper functions ──────────────────────────────────────
def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def index_pdf_to_pinecone(drive_id, topic_id, topic_name):
    try:
        url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return False
        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        full_text = ""
        for page in pdf_doc:
            full_text += page.get_text()
        if len(full_text.strip()) < 100:
            return False
        chunks = chunk_text(full_text)
        vectors = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{topic_id}_{i}".encode()).hexdigest()
            embedding = safe_encode(chunk)
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "chunk_index": i,
                    "text": chunk
                }
            })
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            pinecone_index.upsert(vectors=vectors[i:i + batch_size])
        return True
    except Exception as e:
        print(f"Indexing error: {e}")
        return False

def retrieve_context(query, topic_id, top_k=5, score_threshold=0.3):
    """Returns relevant chunks above the similarity threshold, or empty string."""
    try:
        query_embedding = safe_encode(query)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"topic_id": {"$eq": topic_id}},
            include_metadata=True
        )
        # Filter by relevance score — Pinecone always returns top_k even if unrelated
        relevant = [
            match for match in results["matches"]
            if match.get("score", 0) >= score_threshold
        ]
        chunks = [m["metadata"]["text"] for m in relevant]
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

def is_topic_indexed(topic_id):
    """
    Check indexing by querying with a real topic-related phrase,
    not a dummy string, to avoid false positives.
    """
    try:
        probe = safe_encode(f"drug delivery {topic_id}")
        results = pinecone_index.query(
            vector=probe,
            top_k=1,
            filter={"topic_id": {"$eq": topic_id}},
            include_metadata=True
        )
        matches = results.get("matches", [])
        return len(matches) > 0 and matches[0].get("score", 0) > 0.1
    except:
        return False

# ── Groq call with retry ──────────────────────────────────────
def groq_call_with_retry(messages, system_prompt=None, max_tokens=2048, retries=3, wait=20):
    """
    Calls Groq with exponential backoff on RateLimitError.
    Returns (content_string, error_string).
    max_tokens default raised to 2048 to fit full quiz JSON.
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                messages=full_messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, None
        except RateLimitError as e:
            if attempt < retries - 1:
                sleep_time = wait * (2 ** attempt)  # 20s, 40s, 80s
                with st.spinner(f"⏳ Groq rate limit hit — retrying in {sleep_time}s (attempt {attempt+1}/{retries})..."):
                    time.sleep(sleep_time)
            else:
                return None, "rate_limit"
        except Exception as e:
            print(f"Groq API error: {e}")
            return None, str(e)

# ── Helper functions ──────────────────────────────────────────
def ask_ai(messages, system_prompt):
    content, error = groq_call_with_retry(messages, system_prompt=system_prompt)
    if error == "rate_limit":
        st.warning("⚠️ The AI service is busy. Please wait 30–60 seconds and try again.")
    return content

def ask_ai_with_rag(messages, system_prompt, topic_id=None):
    try:
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        rag_context = ""
        context_found = False

        if topic_id:
            rag_context = retrieve_context(last_user_msg, topic_id, top_k=4)
            context_found = bool(rag_context and len(rag_context.strip()) > 150)

        enhanced_system = system_prompt
        if context_found:
            enhanced_system += f"\n\n## Relevant course material:\n{rag_context}\n\nBase your answer strictly on this material."
        else:
            enhanced_system += (
                "\n\nNo relevant course material was found for this question. "
                "Tell the student clearly that this topic doesn't appear to be covered "
                "in the indexed BE210 lecture for this section, and suggest they check "
                "with their instructor. You may offer a brief general answer but must "
                "flag it as outside the course material."
            )

        content, error = groq_call_with_retry(messages, system_prompt=enhanced_system)
        if error == "rate_limit":
            st.warning("⚠️ The AI service is busy. Please wait 30–60 seconds and try again.")
        return content
    except Exception as e:
        print(f"ask_ai_with_rag error: {e}")
        return None

# ── Student data with session-level cache ─────────────────────
def get_student_data(uid):
    cache_key = f"_student_cache_{uid}"
    cache_time_key = f"_student_cache_time_{uid}"
    now = datetime.datetime.now()
    cached = st.session_state.get(cache_key)
    cached_time = st.session_state.get(cache_time_key)
    if cached and cached_time and (now - cached_time).seconds < 30:
        return cached
    doc = db.collection("students").document(uid).get()
    data = doc.to_dict() if doc.exists else {}
    st.session_state[cache_key] = data
    st.session_state[cache_time_key] = now
    return data

def _bust_student_cache(uid):
    st.session_state.pop(f"_student_cache_{uid}", None)
    st.session_state.pop(f"_student_cache_time_{uid}", None)

def update_student_data(uid, updates):
    db.collection("students").document(uid).update(updates)
    _bust_student_cache(uid)

def firebase_signup(email, password, name):
    try:
        user = auth.create_user(email=email, password=password, display_name=name)
        db.collection("students").document(user.uid).set({
            "name": name,
            "email": email,
            "created_at": datetime.datetime.now(),
            "topics_completed": [],
            "quiz_scores": [],
            "last_login": datetime.datetime.now(),
            "total_time_minutes": 0,
            "review_schedule": {},
            "streak_days": 0,
            "last_study_date": None
        })
        return user.uid, None
    except Exception as e:
        return None, str(e)

def firebase_login(email, password):
    try:
        api_key = st.secrets["firebase"]["api_key"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        r = requests.post(url, json=payload)
        data = r.json()
        if "error" in data:
            return None, None, data["error"]["message"]
        uid = data["localId"]
        user_doc = db.collection("students").document(uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            db.collection("students").document(uid).update({
                "last_login": datetime.datetime.now()
            })
            return uid, user_data, None
        return uid, {}, None
    except Exception as e:
        return None, None, str(e)

def save_quiz_score(uid, topic_id, score, total, passed=False):
    student = get_student_data(uid)
    scores = student.get("quiz_scores", [])
    scores.append({
        "topic": topic_id,
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100),
        "date": datetime.datetime.now().isoformat()
    })
    review_schedule = student.get("review_schedule", {})
    if topic_id not in review_schedule:
        review_schedule[topic_id] = {"interval_index": 0, "next_review": None}
    idx = review_schedule[topic_id].get("interval_index", 0)
    days = REVIEW_INTERVALS[min(idx, len(REVIEW_INTERVALS) - 1)]
    next_review = datetime.datetime.now() + datetime.timedelta(days=days)
    review_schedule[topic_id] = {
        "interval_index": min(idx + 1, len(REVIEW_INTERVALS) - 1),
        "next_review": next_review.isoformat(),
        "last_score": round((score / total) * 100)
    }
    update_student_data(uid, {
        "quiz_scores": scores,
        "review_schedule": review_schedule
    })

def mark_topic_completed(uid, topic_id):
    student = get_student_data(uid)
    completed = student.get("topics_completed", [])
    if topic_id not in completed:
        completed.append(topic_id)
        update_student_data(uid, {"topics_completed": completed})

def update_time_spent(uid, minutes):
    student = get_student_data(uid)
    total = student.get("total_time_minutes", 0) + minutes
    update_student_data(uid, {"total_time_minutes": total})

def get_due_reviews(uid):
    student = get_student_data(uid)
    review_schedule = student.get("review_schedule", {})
    due = []
    now = datetime.datetime.now()
    for topic_id, info in review_schedule.items():
        if info.get("next_review"):
            next_rev = datetime.datetime.fromisoformat(info["next_review"])
            if now >= next_rev:
                due.append(topic_id)
    return due

def generate_quiz(topic_ids, student_data, count=10):
    topics_studied = ", ".join(topic_ids)
    scores = student_data.get("quiz_scores", [])
    recent_scores = [s for s in scores[-5:] if s["topic"] in topic_ids]

    # ── Gather RAG context ────────────────────────────────────
    rag_context = ""
    for tid in topic_ids:
        topic_key = next((k for k, v in TOPICS.items() if v["id"] == tid), None)
        topic_name = TOPICS[topic_key]["description"] if topic_key else tid
        drive_id = TOPICS[topic_key]["drive_id"] if topic_key else None

        context_chunk = retrieve_context(
            f"drug delivery {topic_name} key concepts mechanisms", tid, top_k=12
        )

        # Auto-reindex once if retrieval returned nothing
        if not context_chunk and drive_id:
            st.info(f"🔄 Re-indexing {topic_name} — please wait ~30 seconds...")
            index_pdf_to_pinecone(drive_id, tid, topic_name)
            context_chunk = retrieve_context(
                f"drug delivery {topic_name} key concepts mechanisms", tid, top_k=12
            )

        if context_chunk:
            rag_context += f"\n\n### {topic_name}:\n{context_chunk}"

    # ── Guard: no content = no quiz ───────────────────────────
    if not rag_context or len(rag_context.strip()) < 200:
        st.error(
            "❌ Could not retrieve lecture content. "
            "Please click **'Index this topic for RAG'** on the Learning tab first, then try again."
        )
        return None

    difficulty = "beginner"
    if recent_scores:
        avg = sum(s["percentage"] for s in recent_scores) / len(recent_scores)
        difficulty = "advanced" if avg > 80 else "intermediate" if avg > 60 else "beginner"

    prompt = QUIZ_PROMPT.format(count=count)

    user_content = f"""Generate exactly {count} quiz questions STRICTLY from the lecture material below.
Do NOT invent concepts not present in the material.
Output ONLY the raw JSON object — no markdown, no backticks, no explanation text.

Lecture Material:
-----------------
{rag_context}

Topic: {topics_studied}
Difficulty: {difficulty}
"""

    # ── Call Groq with enough tokens for full quiz JSON ───────
    # 10 questions * ~200 tokens each = ~2000 tokens minimum
    content, error = groq_call_with_retry(
        messages=[{"role": "user", "content": user_content}],
        system_prompt=prompt,
        max_tokens=3000   # raised from 800 — this was the main cause of truncated/broken JSON
    )

    if error == "rate_limit":
        st.error("⚠️ Groq rate limit reached. Please wait 60 seconds and try again.")
        return None
    if not content:
        st.error("⚠️ Quiz generation failed. Please try again.")
        return None

    # ── Parse JSON — strip markdown fences if present ─────────
    raw = content.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers the model sometimes adds
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    # Extract the outermost JSON object as a fallback
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group()

    try:
        data = json.loads(raw)
        # Validate structure
        if "questions" not in data or not isinstance(data["questions"], list):
            st.error("⚠️ Quiz response had unexpected structure. Please try again.")
            return None
        if len(data["questions"]) == 0:
            st.error("⚠️ Quiz returned 0 questions. Please try again.")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}\nRaw:\n{raw[:500]}")
        st.error("⚠️ Could not parse quiz response. Please try again — if this keeps happening, re-index the topic.")
        return None

def check_answer(q, user_ans):
    """Returns True if the answer is correct, including acceptable synonyms for fill_blank."""
    q_type = q.get("type", "mcq")
    correct = q.get("correct", "")
    if q_type == "mcq":
        return str(user_ans) == str(correct)
    elif q_type == "true_false":
        return str(user_ans).lower() == str(correct).lower()
    elif q_type == "fill_blank":
        user = str(user_ans).strip().lower()
        accepted = [str(correct).strip().lower()]
        accepted += [str(a).strip().lower() for a in q.get("acceptable", [])]
        return user in accepted
    elif q_type == "match":
        correct_pairs = q.get("correct_pairs", {})
        if isinstance(user_ans, dict):
            return all(user_ans.get(k) == v for k, v in correct_pairs.items())
        return False
    return False

def get_weak_topics(uid, miss_threshold=3, score_threshold=60):
    student = get_student_data(uid)
    scores = student.get("quiz_scores", [])
    from collections import defaultdict
    topic_fails = defaultdict(list)
    for s in scores:
        if s.get("percentage", 100) < score_threshold:
            topic_fails[s["topic"]].append(s["percentage"])
    weak = {}
    for topic_id, fail_scores in topic_fails.items():
        if len(fail_scores) >= miss_threshold:
            topic_name = next(
                (v["description"] for v in TOPICS.values() if v["id"] == topic_id),
                topic_id
            )
            weak[topic_id] = {
                "name": topic_name,
                "miss_count": len(fail_scores),
                "avg_score": round(sum(fail_scores) / len(fail_scores))
            }
    return weak

def generate_concept_breakdown(topic_id, topic_name):
    rag_context = retrieve_context(
        f"key concepts fundamentals {topic_name}", topic_id, top_k=5
    )
    system = """You are Delivix, a drug delivery tutor. A student keeps getting questions wrong on this topic.
Give a clear, structured concept breakdown covering:
1. The core idea in 2-3 sentences
2. Key terms they must know (with brief definitions)
3. Common misconceptions students have
4. A memorable analogy or example
Be concise but complete. Use simple language."""
    user_content = f"Topic: {topic_name}\n"
    if rag_context:
        user_content += f"\nRelevant lecture material:\n{rag_context}\n"
    user_content += "\nGive a concept breakdown to help this student finally understand this topic."
    return ask_ai([{"role": "user", "content": user_content}], system)

def send_reminder_email(to_email, student_name, due_topics):
    try:
        gmail_user = st.secrets["gmail"]["email"]
        gmail_password = st.secrets["gmail"]["app_password"]
        topic_list = "\n".join([f"• {t}" for t in due_topics])
        body = f"""Hi {student_name}!

Your Delivix AI reminder 💊

It's time to review these topics based on your spaced repetition schedule:

{topic_list}

Log in to Delivix AI to take your review quiz and keep your knowledge fresh!

Remember: The forgetting curve shows we lose ~70% of new info within 24 hours without review.
Regular spaced practice is the key to long-term retention!

Keep learning,
Delivix AI 🔬
"""
        msg = MIMEText(body)
        msg["Subject"] = "🔬 Delivix: Time to Review!"
        msg["From"] = gmail_user
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        return True
    except Exception as e:
        return False

def record_activity():
    st.session_state.last_active = datetime.datetime.now()

def update_active_time():
    if not st.session_state.get("logged_in"):
        return
    now = datetime.datetime.now()
    last = st.session_state.last_active
    diff_mins = (now - last).seconds // 60
    if diff_mins < IDLE_THRESHOLD_MINS:
        elapsed = (now - st.session_state.session_start).seconds // 60
        new_active = elapsed - st.session_state.last_saved_minute
        if new_active > 0:
            st.session_state.active_minutes += new_active
            st.session_state.last_saved_minute = elapsed
            if st.session_state.active_minutes % 5 == 0:
                update_time_spent(st.session_state.uid, 5)

# ── Session state init ────────────────────────────────────────
defaults = {
    "logged_in": False, "uid": None, "user_data": None,
    "messages": [], "challenge_started": False, "selected_challenge": None,
    "learn_messages": {}, "selected_topic": None,
    "quiz_active": False, "quiz_data": None, "quiz_answers": {},
    "quiz_submitted": False, "quiz_topic_ids": [],
    "session_start": datetime.datetime.now(), "auth_mode": "login",
    "last_active": datetime.datetime.now(),
    "active_minutes": 0,
    "last_saved_minute": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 💊 Delivix AI")
        st.markdown("*BE210 Drug Delivery — IISc*")
        st.divider()

        mode = st.radio("Select mode", ["Login", "Sign Up"], horizontal=True, key="auth_mode_radio")

        if mode == "Login":
            st.subheader("Welcome back!")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", type="primary", use_container_width=True):
                if email and password:
                    with st.spinner("Logging in..."):
                        uid, user_data, error = firebase_login(email, password)
                    if error:
                        st.error(f"Login failed: {error}")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.uid = uid
                        st.session_state.user_data = user_data
                        st.session_state.session_start = datetime.datetime.now()
                        st.session_state.last_active = datetime.datetime.now()
                        st.rerun()
                else:
                    st.warning("Please enter email and password.")

        else:
            st.subheader("Create your account")
            name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password (min 6 chars)", type="password", key="signup_pass")
            if st.button("Create Account", type="primary", use_container_width=True):
                if name and email and password:
                    with st.spinner("Creating account..."):
                        uid, error = firebase_signup(email, password, name)
                    if error:
                        st.error(f"Signup failed: {error}")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.uid = uid
                        st.session_state.user_data = {
                            "name": name,
                            "email": email,
                            "topics_completed": [],
                            "quiz_scores": [],
                            "total_time_minutes": 0,
                            "review_schedule": {}
                        }
                        st.session_state.session_start = datetime.datetime.now()
                        st.session_state.last_active = datetime.datetime.now()
                        st.rerun()
                else:
                    st.warning("Please fill all fields.")

# ══════════════════════════════════════════════════════════════
# MAIN APP (logged in)
# ══════════════════════════════════════════════════════════════
else:
    update_active_time()

    student = get_student_data(st.session_state.uid)
    student_name = student.get("name", "Student")
    due_reviews = get_due_reviews(st.session_state.uid)

    # ── Top bar ───────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"### 💊 Delivix AI &nbsp; | &nbsp; 👋 {student_name}")
    with col2:
        topics_done = len(student.get("topics_completed", []))
        st.metric("Topics Completed", f"{topics_done}/10")
        progress = topics_done / 10
        st.progress(progress)
        st.caption(f"{topics_done}/10 lectures mastered 🧠")
    with col3:
        if st.button("Logout"):
            remaining = st.session_state.active_minutes % 5
            if remaining > 0:
                update_time_spent(st.session_state.uid, remaining)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Review due banner ─────────────────────────────────────
    if due_reviews:
        topic_names = [k for k, v in TOPICS.items() if v["id"] in due_reviews]
        st.warning(f"⏰ **Time to review!** Spaced repetition due for: {', '.join(topic_names)}")
        if st.button("📝 Take Review Quiz Now"):
            st.session_state.quiz_active = True
            st.session_state.quiz_topic_ids = due_reviews
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

    st.divider()

    # ── Weak topic warning ────────────────────────────────────
    weak_topics = get_weak_topics(st.session_state.uid)
    if weak_topics:
        for topic_id, info in weak_topics.items():
            st.error(
                f"⚠️ **Struggling with {info['name'].split(' - ')[-1]}** — "
                f"you've scored below 60% on this topic {info['miss_count']} times "
                f"(avg: {info['avg_score']}%)"
            )
        with st.expander("📖 Get a concept breakdown for your weak topics"):
            for topic_id, info in weak_topics.items():
                st.markdown(f"### {info['name'].split(' - ')[-1]}")
                breakdown_key = f"breakdown_{topic_id}"
                if breakdown_key not in st.session_state:
                    st.session_state[breakdown_key] = None
                if st.session_state[breakdown_key] is None:
                    if st.button(f"Explain this topic to me", key=f"btn_breakdown_{topic_id}"):
                        with st.spinner("Generating concept breakdown..."):
                            st.session_state[breakdown_key] = generate_concept_breakdown(
                                topic_id, info["name"]
                            )
                        st.rerun()
                else:
                    st.markdown(st.session_state[breakdown_key])
                    if st.button("🔄 Regenerate", key=f"regen_{topic_id}"):
                        st.session_state[breakdown_key] = None
                        st.rerun()
                st.divider()

    # ── QUIZ MODE ─────────────────────────────────────────────
    if st.session_state.quiz_active:
        st.subheader("📝 Spaced Repetition Quiz")
        st.caption("Based on your forgetting curve schedule")

        if not st.session_state.quiz_data:
            with st.spinner("Generating personalized quiz..."):
                quiz = generate_quiz(st.session_state.quiz_topic_ids, student)
            if quiz:
                st.session_state.quiz_data = quiz
            else:
                if st.button("← Back"):
                    st.session_state.quiz_active = False
                    st.rerun()

        if st.session_state.quiz_data:
            questions = st.session_state.quiz_data.get("questions", [])

            if not st.session_state.quiz_submitted:
                for i, q in enumerate(questions):
                    q_type = q.get("type", "mcq")
                    st.markdown(f"**Q{i+1}: {q['question']}**")

                    if q_type == "mcq":
                        answer = st.radio("", q["options"], key=f"q{i}", label_visibility="collapsed")
                        if answer:
                            st.session_state.quiz_answers[i] = answer[0]

                    elif q_type == "true_false":
                        answer = st.radio("", ["True", "False"], key=f"q{i}", label_visibility="collapsed")
                        if answer:
                            st.session_state.quiz_answers[i] = answer

                    elif q_type == "fill_blank":
                        answer = st.text_input("Your answer:", key=f"q{i}")
                        if answer:
                            st.session_state.quiz_answers[i] = answer.strip().lower()

                    elif q_type == "match":
                        st.caption("Match each term with the correct definition:")
                        left_items = q.get("left", [])
                        right_items = q.get("right", [])
                        match_answers = {}
                        for term in left_items:
                            choice = st.selectbox(f"{term} →", right_items, key=f"q{i}_{term}")
                            match_answers[term] = choice
                        if match_answers:
                            st.session_state.quiz_answers[i] = match_answers

                    st.markdown("")

                if st.button("Submit Quiz", type="primary"):
                    record_activity()
                    st.session_state.quiz_submitted = True
                    st.rerun()

            else:
                score = 0
                for i, q in enumerate(questions):
                    user_ans = st.session_state.quiz_answers.get(i, "")
                    is_correct = check_answer(q, user_ans)
                    if is_correct:
                        score += 1
                    icon = "✅" if is_correct else "❌"
                    st.markdown(f"{icon} **Q{i+1}: {q['question']}**")
                    if q.get("type") == "match":
                        st.markdown(f"Correct pairs: **{q.get('correct_pairs', {})}**")
                    else:
                        st.markdown(f"Your answer: **{user_ans}** | Correct: **{q.get('correct', '')}**")
                    st.caption(f"💡 {q['explanation']}")
                    st.markdown("")

                percentage = round((score / len(questions)) * 100)
                st.divider()
                st.markdown(f"### Score: {score}/{len(questions)} ({percentage}%)")

                if percentage >= 80:
                    st.success("🎉 Excellent! Knowledge well retained.")
                elif percentage >= 60:
                    st.info("👍 Good work! A bit more review will help.")
                else:
                    st.warning("📚 Review these topics again — revisit the Learning tab.")

                for topic_id in st.session_state.quiz_topic_ids:
                    save_quiz_score(st.session_state.uid, topic_id, score, len(questions), passed=(percentage >= 70))

                next_days = REVIEW_INTERVALS[min(1, len(REVIEW_INTERVALS) - 1)]
                st.info(f"📅 Next review scheduled in **{next_days} days** (spaced repetition)")

                if st.button("✅ Done", type="primary"):
                    st.session_state.quiz_active = False
                    st.session_state.quiz_data = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()

    # ── MAIN TABS ─────────────────────────────────────────────
    else:
        tab1, tab2, tab3 = st.tabs(["📚 Learning", "🧪 Test Yourself", "📊 My Progress"])

        # ════════════════════════════════════════════════════
        # TAB 1 — LEARNING
        # ════════════════════════════════════════════════════
        with tab1:
            st.subheader("📚 BE210 Lecture Material")
            st.caption("Read the lecture slides and ask Delivix any questions")

            topic_name = st.selectbox("Choose a topic:", list(TOPICS.keys()), key="topic_selector")
            topic = TOPICS[topic_name]
            topic_id = topic["id"]

            st.markdown(f"*{topic['description']}*")

            # ── RAG indexing status ───────────────────────
            drive_id = topic["drive_id"]
            if not is_topic_indexed(topic_id):
                st.warning("⚡ This topic hasn't been indexed for AI yet.")
                if st.button("🔍 Index this topic for RAG", key=f"index_{topic_id}"):
                    with st.spinner("Reading and indexing PDF... this takes ~30 seconds"):
                        success = index_pdf_to_pinecone(drive_id, topic_id, topic_name)
                    if success:
                        st.success("✅ Indexed! The AI can now answer from the actual lecture content.")
                        st.rerun()
                    else:
                        st.error("Indexing failed. Make sure the PDF is publicly shared on Drive.")
            else:
                st.success("✅ RAG active — AI answers are grounded in your lecture material")

            # Display PDF via Google Drive
            pdf_display = f'<iframe src="https://drive.google.com/file/d/{drive_id}/preview" width="100%" height="700px" style="border:none;border-radius:8px;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

            # ── Quiz-based completion ─────────────────────
            completed = student.get("topics_completed", [])
            if topic_id not in completed:
                st.divider()
                st.markdown("### 📝 Completion Quiz")
                st.caption("Read the material above, then take this short quiz to mark the topic as complete. You need 7/10 to pass.")

                comp_quiz_key = f"comp_quiz_{topic_id}"
                comp_ans_key  = f"comp_ans_{topic_id}"
                comp_sub_key  = f"comp_submitted_{topic_id}"

                if comp_quiz_key not in st.session_state:
                    st.session_state[comp_quiz_key] = None
                if comp_ans_key not in st.session_state:
                    st.session_state[comp_ans_key] = {}
                if comp_sub_key not in st.session_state:
                    st.session_state[comp_sub_key] = False

                # PHASE 1: Start
                if st.session_state[comp_quiz_key] is None:
                    if st.button("📖 I've finished reading — Take Quiz", type="primary", key=f"start_comp_{topic_id}"):
                        record_activity()
                        with st.spinner("Generating quiz..."):
                            quiz = generate_quiz([topic_id], student)
                        if quiz:
                            questions = quiz.get("questions", [])[:10]
                            st.session_state[comp_quiz_key] = {"questions": questions}
                            st.session_state[comp_sub_key] = False
                            st.session_state[comp_ans_key] = {}
                            st.rerun()
                        # generate_quiz() already shows the error message

                # PHASE 2: Answer questions
                elif not st.session_state[comp_sub_key]:
                    questions = st.session_state[comp_quiz_key]["questions"]
                    for i, q in enumerate(questions):
                        q_type = q.get("type", "mcq")
                        st.markdown(f"**Q{i+1}: {q['question']}**")

                        if q_type == "mcq":
                            answer = st.radio("", q["options"], key=f"comp_q{topic_id}_{i}", label_visibility="collapsed")
                            if answer:
                                st.session_state[comp_ans_key][i] = answer[0]

                        elif q_type == "true_false":
                            answer = st.radio("", ["True", "False"], key=f"comp_q{topic_id}_{i}", label_visibility="collapsed")
                            if answer:
                                st.session_state[comp_ans_key][i] = answer

                        elif q_type == "fill_blank":
                            answer = st.text_input("Your answer:", key=f"comp_q{topic_id}_{i}")
                            if answer:
                                st.session_state[comp_ans_key][i] = answer.strip().lower()

                        elif q_type == "match":
                            st.caption("Match each term with the correct definition:")
                            left_items = q.get("left", [])
                            right_items = q.get("right", [])
                            match_answers = {}
                            for term in left_items:
                                choice = st.selectbox(f"{term} →", right_items, key=f"comp_q{topic_id}_{i}_{term}")
                                match_answers[term] = choice
                            if match_answers:
                                st.session_state[comp_ans_key][i] = match_answers

                        st.markdown("")

                    if st.button("Submit", type="primary", key=f"submit_comp_{topic_id}"):
                        record_activity()
                        st.session_state[comp_sub_key] = True
                        st.rerun()

                # PHASE 3: Results
                else:
                    questions = st.session_state[comp_quiz_key]["questions"]
                    score = 0
                    for i, q in enumerate(questions):
                        user_ans = st.session_state[comp_ans_key].get(i, "")
                        is_correct = check_answer(q, user_ans)
                        if is_correct:
                            score += 1
                        icon = "✅" if is_correct else "❌"
                        st.markdown(f"{icon} **Q{i+1}: {q['question']}**")
                        st.caption(f"💡 {q['explanation']}")

                    st.divider()
                    passed = score >= 7
                    # Save score once here only (removed duplicate call that existed before)
                    save_quiz_score(st.session_state.uid, topic_id, score, 10, passed=passed)

                    if passed:
                        st.success(f"🎉 Passed! {score}/10 — Topic unlocked!")
                        st.balloons()
                        mark_topic_completed(st.session_state.uid, topic_id)
                        st.rerun()
                    else:
                        st.error(f"❌ {score}/10 — You need 7/10 to pass. Re-read the material and try again.")
                        if st.button("🔄 Try Again", key=f"retry_comp_{topic_id}"):
                            st.session_state[comp_quiz_key] = None
                            st.session_state[comp_ans_key] = {}
                            st.session_state[comp_sub_key] = False
                            st.rerun()
            else:
                st.success("✅ You've completed this topic!")

            st.divider()
            st.subheader("💬 Ask Delivix about this topic")

            if topic_id not in st.session_state.learn_messages:
                st.session_state.learn_messages[topic_id] = []

            for msg in st.session_state.learn_messages[topic_id]:
                with st.chat_message(msg["role"], avatar="🔬" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

            learn_input = st.chat_input(f"Ask about {topic_name}...")
            if learn_input:
                record_activity()
                msgs = st.session_state.learn_messages[topic_id]
                msgs.append({"role": "user", "content": learn_input})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(learn_input)
                context = f"\nThe student is studying: {topic_name} ({topic['description']})\nBE210 IISc Drug Delivery course."
                with st.chat_message("assistant", avatar="🔬"):
                    with st.spinner("Thinking..."):
                        reply = ask_ai_with_rag(msgs, LEARN_PROMPT + context, topic_id=topic_id)
                    st.markdown(reply)
                msgs.append({"role": "assistant", "content": reply})

            if st.session_state.learn_messages.get(topic_id):
                if st.button("🗑️ Clear Chat", key=f"clear_{topic_id}"):
                    st.session_state.learn_messages[topic_id] = []
                    st.rerun()

        # ════════════════════════════════════════════════════
        # TAB 2 — TEST YOURSELF
        # ════════════════════════════════════════════════════
        with tab2:
            st.subheader("🧪 Test Yourself")
            st.caption("Pick a clinical challenge and let Delivix coach you through it")

            CHALLENGES = {
                "🧬 Liver Gene Therapy": "Deliver a gene therapy to the liver without systemic toxicity",
                "🧠 Brain Tumor siRNA": "Silence an oncogene in glioblastoma crossing the blood-brain barrier",
                "💉 mRNA Vaccine": "Design an mRNA vaccine platform for rapid pandemic response",
                "💊 Oral Peptide Delivery": "Deliver insulin orally across the GI tract without degradation",
                "⚗️ Custom Challenge": None
            }

            if not st.session_state.challenge_started:
                selected = st.radio("Choose your challenge:", list(CHALLENGES.keys()))
                custom_text = ""
                if selected == "⚗️ Custom Challenge":
                    custom_text = st.text_area("Describe your clinical problem:", height=80)

                if st.button("🚀 Start Challenge", type="primary", use_container_width=True):
                    challenge_desc = custom_text if selected == "⚗️ Custom Challenge" else CHALLENGES[selected]
                    if selected == "⚗️ Custom Challenge" and not custom_text.strip():
                        st.warning("Please describe your challenge.")
                    else:
                        record_activity()
                        st.session_state.selected_challenge = challenge_desc
                        st.session_state.challenge_started = True
                        init_prompt = f"""Student chose: "{challenge_desc}"
Greet as Delivix, acknowledge clinical importance briefly, ask your FIRST probing question. No answers yet."""
                        with st.spinner("Delivix is preparing..."):
                            opening = ask_ai([{"role": "user", "content": init_prompt}], COACH_PROMPT)
                        st.session_state.messages.append({"role": "assistant", "content": opening})
                        st.rerun()
            else:
                st.info(f"**Challenge:** {st.session_state.selected_challenge}")
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"], avatar="🔬" if msg["role"] == "assistant" else "👤"):
                        st.markdown(msg["content"])

                user_input = st.chat_input("Share your thinking...")
                if user_input:
                    record_activity()
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(user_input)
                    with st.chat_message("assistant", avatar="🔬"):
                        with st.spinner("Thinking..."):
                            reply = ask_ai(st.session_state.messages, COACH_PROMPT)
                        st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                st.divider()
                if st.button("🔄 New Challenge"):
                    st.session_state.messages = []
                    st.session_state.challenge_started = False
                    st.session_state.selected_challenge = None
                    st.rerun()

        # ════════════════════════════════════════════════════
        # TAB 3 — PROGRESS
        # ════════════════════════════════════════════════════
        with tab3:
            st.subheader("📊 My Learning Progress")
            student = get_student_data(st.session_state.uid)

            col1, col2, col3, col4 = st.columns(4)
            completed = student.get("topics_completed", [])
            scores = student.get("quiz_scores", [])
            total_mins = student.get("total_time_minutes", 0)
            avg_score = round(sum(s["percentage"] for s in scores) / len(scores)) if scores else 0

            col1.metric("Topics Completed", f"{len(completed)}/10")
            col2.metric("Quizzes Taken", len(scores))
            col3.metric("Avg Quiz Score", f"{avg_score}%")
            col4.metric("Time Studied", f"{total_mins} min")

            st.divider()

            st.markdown("#### 📋 Topic Status")
            for topic_name, topic_info in TOPICS.items():
                tid = topic_info["id"]
                is_done = tid in completed
                review_schedule = student.get("review_schedule", {})
                next_review = review_schedule.get(tid, {}).get("next_review")

                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    icon = "✅" if is_done else "⭕"
                    st.markdown(f"{icon} **{topic_name}**")
                with col2:
                    last_score = review_schedule.get(tid, {}).get("last_score", "-")
                    st.markdown(f"Last: **{last_score}%**" if last_score != "-" else "Not tested")
                with col3:
                    if next_review:
                        nrd = datetime.datetime.fromisoformat(next_review)
                        days_left = (nrd - datetime.datetime.now()).days
                        if days_left <= 0:
                            st.markdown("🔴 **Review due now!**")
                        else:
                            st.markdown(f"📅 Review in {days_left} days")
                    else:
                        st.markdown("No review scheduled")

            st.divider()

            if scores:
                st.markdown("#### 📈 Quiz History")
                for s in reversed(scores[-10:]):
                    date = s["date"][:10]
                    pct = s["percentage"]
                    bar = "🟩" * (pct // 20) + "⬜" * (5 - pct // 20)
                    st.markdown(f"`{date}` | {s['topic']} | {bar} **{pct}%**")

            st.divider()

            st.markdown("#### 🧠 Your Spaced Repetition Schedule")
            st.caption("Based on Ebbinghaus forgetting curve — review at optimal intervals to maximize retention")
            intervals_str = " → ".join([f"{d}d" for d in REVIEW_INTERVALS])
            st.info(f"Review intervals: **{intervals_str}**")

            st.divider()

            st.markdown("#### 📧 Email Reminders")
            if st.button("📬 Send Me a Reminder Email Now"):
                due = get_due_reviews(st.session_state.uid)
                topic_names = [k for k, v in TOPICS.items() if v["id"] in due] if due else ["All topics — stay sharp!"]
                success = send_reminder_email(
                    student.get("email", ""),
                    student_name,
                    topic_names
                )
                if success:
                    st.success("Reminder email sent!")
                else:
                    st.error("Could not send email. Check Gmail settings in secrets.toml.")
