import os
import base64
import mimetypes
from io import BytesIO
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ---------- Load env & constants ----------
load_dotenv()

API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar-pro"


# ---------- State management ----------
def init_state(model=None):
    defaults = {
        "started": False,
        "imagename": None,
        "imagebytes": None,
        "imagemime": None,
        "imagedatauri": None,
        "displaymessages": [],
        "apihistory": [],
        "mode": "Auto",
        "contextnotes": "",
        "specimenlabel": "",
        "model": DEFAULT_MODEL,
        "lastuploadedsignature": None,
        "studentobservations": "",
        "studentbestanswer": "",
        "knownname": "",
        "studentname": "",
        "includeautozoom": False,
        "zoomfraction": 0.5,
        "authenticated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if model is not None:
        st.session_state["model"] = model


def login_screen():
    st.title("GIA – Guided Image Analysis")
    st.caption("Instructor pilot: enter the access password to use this app.")

    pw = st.text_input("Access password", type="password")
    correct_pw = os.getenv("APP_PASSWORD", "").strip()

    if not correct_pw:
        st.error("APP_PASSWORD is not set in your .env file.")
    elif pw == correct_pw:
        st.session_state.authenticated = True
        st.success("Logged in. Loading the app...")
        st.rerun()
    elif pw:
        st.error("Incorrect password. Please check with your instructor.")


def reset_app():
    keep_model = st.session_state.get("model", DEFAULT_MODEL)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state(model=keep_model)


def get_api_key():
    return os.getenv("PERPLEXITY_API_KEY")


# ---------- Image utilities ----------
def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type
    if not mime:
        mime = mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return raw, mime, data_uri


def update_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return

    signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.lastuploadedsignature == signature:
        return

    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.imagename = uploaded_file.name
    st.session_state.imagebytes = raw
    st.session_state.imagemime = mime
    st.session_state.imagedatauri = data_uri
    st.session_state.lastuploadedsignature = signature


def get_image_contents_for_api():
    """
    Returns a list of image content dicts for the API:
    - Always includes the full image.
    - Optionally includes a zoomed center crop if enabled.
    """
    contents = []
    if not st.session_state.imagebytes or not st.session_state.imagedatauri:
        return contents

    # Full image
    contents.append(
        {
            "type": "image_url",
            "image_url": {"url": st.session_state.imagedatauri},
        }
    )

    if not st.session_state.includeautozoom:
        return contents

    try:
        img = Image.open(BytesIO(st.session_state.imagebytes))
        w, h = img.size
        frac = st.session_state.zoomfraction
        frac = max(0.1, min(frac, 1.0))
        cw, ch = int(w * frac), int(h * frac)
        left = (w - cw) // 2
        top = (h - ch) // 2
        right = left + cw
        bottom = top + ch
        cropcenter = img.crop((left, top, right, bottom))

        buf = BytesIO()
        fmt = img.format if img.format in ("JPEG", "PNG", "WEBP") else "PNG"
        cropcenter.save(buf, format=fmt)
        cropbytes = buf.getvalue()

        b64 = base64.b64encode(cropbytes).decode("utf-8")
        mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(fmt, "image/png")
        cropdatauri = f"data:{mime};base64,{b64}"

        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": cropdatauri},
            }
        )
    except Exception:
        pass

    return contents


# ---------- Prompt building ----------
def build_system_prompt(mode):
    mode_guidance = {
        "Auto": (
            "Decide which domain best fits the specimen: rock, mineral, fossil, "
            "sand/granular sediment, soil, or forensic particulate. If the domain "
            "is unclear, say so explicitly and explain what visible evidence would help."
        ),
        "Rock": (
            "Focus on rock identification. Prioritize texture, grain size, "
            "clast vs crystalline texture, sorting, rounding, layering, vesicles, "
            "foliation, cement, and matrix. Avoid overclaiming composition when the "
            "image does not support it."
        ),
        "Mineral": (
            "Focus on mineral identification. Prioritize color, luster, transparency, "
            "habit, cleavage/fracture clues, crystal form, and likely hardness implications "
            "if visible. Avoid claiming a mineral species unless the image evidence is strong."
        ),
        "Fossil": (
            "Focus on fossil identification. Prioritize symmetry, segmentation, ornamentation, "
            "shell geometry, visible structures, preservation style, and likely fossil group. "
            "Avoid forcing a genus/species ID from weak evidence."
        ),
        "SandGranular": (
            "Focus on sand, grains, sediment, soil particles, or particulate forensic-style "
            "material. Comment on grain size class, sorting, roundness/angularity, sphericity, "
            "transparency/opacity, luster, quartz likelihood, feldspar clues, lithic fragments, "
            "heavy minerals, organic fragments, and what cannot be determined from this image alone. "
            "Do not call it a powder or crystal substance unless the image clearly supports that language."
        ),
        "Forensic": (
            "Focus on trace material or forensic-style particulate evidence. Describe visible "
            "particle classes, shape variation, color variation, transparency, possible natural "
            "vs manufactured particles, contamination risk, and what follow-up observations are "
            "needed before any strong claim. Be especially conservative."
        ),
    }

    domain_text = mode_guidance.get(mode, mode_guidance["Auto"]).strip()

    return f"""
You are a conversational geology tutor for an introductory college teaching app in 2026.

General rules:
- Sound like a patient lab instructor, not a script. Vary your wording and examples.
- Distinguish direct observation from interpretation and keep observations honest, even if they do not fully support the instructor’s known name.
- Be useful, specific, cautious, and friendly. Do not overclaim.
- When discussing geology, focus on the actual descriptive features students should observe.
- Teach the student how to look, not just what to conclude.
- If the evidence is weak, offer a small number of plausible interpretations and explain why.
- Keep responses compact (about 4–8 sentences) and clearly tied to THIS particular image and chat turn.
- Whenever it helps, explicitly connect your explanation to what the student just said.
- Avoid repeating the same examples or sentence openings you used earlier in this conversation.
- Use the full image for overall context and scale, and any zoomed images to inspect fine details like textures, grain boundaries, cleavage, or fossils.
- If the student asks for a summary or evaluation, provide it in a friendly, concise way that validates what they did well and gives specific next steps.

Response style:
- Answer in 1–2 natural-sounding paragraphs.
- Finish with exactly one short, open-ended question to keep the conversation going unless the student explicitly asks for a summary or says they are done.

Domain instructions:
{domain_text}
""".strip()


def build_api_messages(messages=None):
    system_content = build_system_prompt(st.session_state.mode)
    messages_out = [
        {"role": "system", "content": system_content},
    ]

    for item in st.session_state.apihistory:
        if item["role"] == "user":
            content = [{"type": "text", "text": item["content"]}]
            images = get_image_contents_for_api()
            content.extend(images)
            messages_out.append({"role": "user", "content": content})
        else:
            messages_out.append({"role": "assistant", "content": item["content"]})

    return messages_out


def call_perplexity(messages=None):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if messages is None:
        messages = build_api_messages()

    payload = {
        "model": st.session_state.model,
        "messages": messages,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Perplexity error {response.status_code}: {response.text[:2000]}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------- Conversation flows ----------
def start_first_analysis():
    if not st.session_state.imagedatauri:
        raise RuntimeError("Please upload an image first.")

    label = st.session_state.specimenlabel.strip() or "No specimen label provided"
    notes = st.session_state.contextnotes.strip() or "No additional notes provided"
    studentname = st.session_state.studentname.strip() or "no name provided"
    observations = st.session_state.studentobservations.strip() or "none entered yet"
    bestanswer = st.session_state.studentbestanswer.strip() or "none entered yet"
    knownname = st.session_state.knownname.strip() or "none provided"

    starterprompt = f"""
Please analyze the uploaded specimen image for a teaching app.

Selected mode: {st.session_state.mode}
Specimen label: {label}
Student/instructor notes: {notes}
Student name (if given, use occasionally in a natural, non-repetitive way): {studentname}
Student observations so far: {observations}
Student best answer so far: {bestanswer}
Known name from instructor (if any): {knownname}

Your job:
- Start with observation before interpretation.
- Clearly distinguish what is directly visible from interpretive inference.
- If this is sand or granular material, explicitly address whether the visible grains appear well sorted or poorly sorted, whether quartz is likely, whether lithic grains may be present, and what cannot be determined confidently.
- If the evidence does not support a strong ID, say so clearly.
- Sound conversational and non-repetitive, as if you are talking with the student at the lab bench.
- Use the full image for scale and any zoomed images to inspect textures and fine details.
- End with exactly one open-ended question that invites the student to make or refine an observation.
""".strip()

    visibleusertext = (
        "Please analyze this uploaded specimen. "
        f"Mode: {st.session_state.mode}. "
        f"Label: {label}. "
        f"Notes: {notes}."
    )

    st.session_state.apihistory.append({"role": "user", "content": starterprompt})
    st.session_state.displaymessages.append({"role": "user", "content": visibleusertext})

    reply = call_perplexity()
    st.session_state.apihistory.append({"role": "assistant", "content": reply})
    st.session_state.displaymessages.append({"role": "assistant", "content": reply})
    st.session_state.started = True


def send_followup(usertext: str):
    usertext = usertext.strip()
    if not usertext:
        return

    studentname = st.session_state.studentname.strip() or "no name provided"
    observations = st.session_state.studentobservations.strip() or "none entered yet"
    bestanswer = st.session_state.studentbestanswer.strip() or "none entered yet"
    knownname = st.session_state.knownname.strip() or "none provided"

    followupprompt = f"""
Student follow-up: {usertext}

Context:
- Student name (mention naturally at most once per reply): {studentname}
- Mode: {st.session_state.mode}
- Specimen label: {st.session_state.specimenlabel or "none"}
- Student observations: {observations}
- Student best answer: {bestanswer}
- Known name from instructor: {knownname}

Your job:
- Answer as a conversational geology tutor.
- Stay grounded in the uploaded image and the student's words.
- If the student provides new observations or corrections, incorporate them honestly.
- If new information or the known name conflicts with your earlier idea, politely explain the mismatch and keep your observations honest to the image.
- Be concise (about 4–8 sentences), supportive, and vary your phrasing so it does not sound like a template.
- When helpful, refer the student to specific parts of the main image or the zoomed view (e.g., "look closely at the zoomed image where the grains touch").
- If the student asks for a summary or evaluation, provide it without a follow-up question.
- Otherwise, end with exactly one open-ended question that nudges the student toward a next observation or comparison.
""".strip()

    st.session_state.displaymessages.append({"role": "user", "content": usertext})
    st.session_state.apihistory.append({"role": "user", "content": followupprompt})
    reply = call_perplexity()
    st.session_state.apihistory.append({"role": "assistant", "content": reply})
    st.session_state.displaymessages.append({"role": "assistant", "content": reply})


def save_conversation_to_file():
    if not st.session_state.displaymessages:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    studentname = st.session_state.studentname.strip() or "student"
    safename = "".join(c if c.isalnum() else "_" for c in studentname)
    filename = f"GIA_conversation_{safename}_{timestamp}.txt"

    lines = []
    lines.append("-" * 60)
    lines.append("GIA Guided Image Analysis - Conversation Log")
    lines.append("-" * 60)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Student: {st.session_state.studentname or 'not provided'}")
    lines.append(f"Specimen label: {st.session_state.specimenlabel or 'none'}")
    lines.append(f"Mode: {st.session_state.mode}")
    lines.append(f"Known name: {st.session_state.knownname or 'none'}")
    lines.append("-" * 60)
    lines.append("")

    for msg in st.session_state.displaymessages:
        role = "STUDENT" if msg["role"] == "user" else "AI TUTOR"
        lines.append(role)
        lines.append(msg["content"])
        lines.append("")

    lines.append("-" * 60)
    lines.append("End of conversation")
    lines.append("-" * 60)

    content = "\n".join(lines)
    return filename, content


# ---------- Streamlit page config & geology theme ----------
st.set_page_config(
    page_title="GIA – Guided Image Analysis",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --gia-bg: #F7F3EE;
        --gia-primary: #5D4037;
        --gia-secondary: #8D6E63;
        --gia-accent: #00796B;
        --gia-text: #2B2B2B;
        --gia-muted: #6B6B73;
    }

    .main {
        background: radial-gradient(circle at top left, #E0F2F1 0, #F7F3EE 40%, #FFFFFF 100%);
    }

    .gia-header {
        text-align: center;
        padding-top: 0.4rem;
        padding-bottom: 0.1rem;
        color: var(--gia-primary);
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    .gia-subtitle {
        text-align: center;
        color: var(--gia-muted);
        font-size: 0.95rem;
        margin-bottom: 1.0rem;
    }

    .gia-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background-color: rgba(93, 64, 55, 0.08);
        color: var(--gia-accent);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .gia-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 1.0rem 1.2rem;
        border: 1px solid rgba(93, 64, 55, 0.15);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.0rem;
    }

    .gia-card h3 {
        margin-top: 0;
        color: var(--gia-primary);
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    .gia-body-text {
        color: var(--gia-text);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .gia-helper-text {
        color: var(--gia-muted);
        font-size: 0.85rem;
    }

    .gia-separator {
        text-align: center;
        color: var(--gia-secondary);
        font-size: 1.2rem;
        margin: 0.5rem 0 0.8rem 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EFEBE9 0, #F7F3EE 40%, #FFFFFF 100%);
        border-right: 1px solid rgba(93, 64, 55, 0.18);
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, var(--gia-primary), var(--gia-accent));
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.5rem 1.2rem;
        font-size: 0.95rem;
        font-weight: 600;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
        box-shadow: 0 3px 9px rgba(0, 0, 0, 0.16);
        cursor: pointer;
        transition: transform 0.06s ease-in-out, box-shadow 0.06s ease-in-out, filter 0.06s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 5px 13px rgba(0, 0, 0, 0.22);
    }

    div.stButton > button:first-child:active {
        transform: translateY(0px);
        box-shadow: 0 3px 7px rgba(0, 0, 0, 0.18);
    }

    .stChatMessage .markdown-text-container p {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Init state & login ----------
init_state()

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# ---------- Header ----------
st.markdown(
    """
    <h1 class="gia-header">GIA – Guided Image Analysis 🪨</h1>
    <div class="gia-subtitle">
        <span class="gia-badge">Intro Geology Lab</span>
        <br/><br/>
        Upload a specimen image, then work with the tutor to practice observation and interpretation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="gia-separator">🧪 ⛰️ 🧪</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Settings")

    st.session_state.model = st.text_input(
        "Perplexity model",
        value=st.session_state.model,
    )

    st.session_state.mode = st.selectbox(
        "Specimen mode",
        ["Auto", "Rock", "Mineral", "Fossil", "SandGranular", "Forensic"],
        index=["Auto", "Rock", "Mineral", "Fossil", "SandGranular", "Forensic"].index(
            st.session_state.mode
        )
        if st.session_state.mode in ["Auto", "Rock", "Mineral", "Fossil", "SandGranular", "Forensic"]
        else 0,
    )

    st.session_state.specimenlabel = st.text_input(
        "Specimen label (sample ID)",
        value=st.session_state.specimenlabel,
        placeholder="e.g., Beach sand sample A",
    )

    st.session_state.contextnotes = st.textarea(
        "Context notes",
        value=st.session_state.contextnotes,
        height=120,
        placeholder="e.g., beach sample, hand lens view, no scale bar, bright overhead light",
    )

    st.session_state.studentname = st.text_input(
        "Your name (optional, for the tutor)",
        value=st.session_state.studentname,
        placeholder="e.g., Alex",
    )

    st.markdown("---")
    st.subheader("Image zoom options")

    st.session_state.includeautozoom = st.checkbox(
        "Include a center zoom image for the AI",
        value=st.session_state.includeautozoom,
        help="Sends a zoomed-in crop along with the full image so the AI can inspect textures more closely.",
    )

    st.session_state.zoomfraction = st.slider(
        "Zoom size (fraction of image)",
        min_value=0.2,
        max_value=0.8,
        value=float(st.session_state.zoomfraction),
        step=0.1,
        help="Controls how large the center crop is relative to the full image.",
    )

    st.markdown("---")
    if st.button("Reset app", use_container_width=True):
        reset_app()
        st.rerun()

# ---------- Main layout ----------
left, right = st.columns([1, 1.2])

with left:
    st.markdown('<div class="gia-card">', unsafe_allow_html=True)
    st.markdown("<h3>Specimen image</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload specimen image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        update_uploaded_image(uploaded_file)
        st.image(
            st.session_state.imagebytes,
            caption=st.session_state.imagename,
            use_container_width=True,
        )

        if st.session_state.includeautozoom and st.session_state.imagebytes:
            try:
                img = Image.open(BytesIO(st.session_state.imagebytes))
                w, h = img.size
                frac = st.session_state.zoomfraction
                frac = max(0.1, min(frac, 1.0))
                cw, ch = int(w * frac), int(h * frac)
                leftcrop = (w - cw) // 2
                topcrop = (h - ch) // 2
                rightcrop = leftcrop + cw
                bottomcrop = topcrop + ch
                cropcenter = img.crop((leftcrop, topcrop, rightcrop, bottomcrop))
                st.image(
                    cropcenter,
                    caption=f"Auto zoom center ~{int(frac * 100)}% of image",
                    use_container_width=True,
                )
            except Exception:
                pass

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gia-card">', unsafe_allow_html=True)
    st.markdown("<h3>Student input (optional)</h3>", unsafe_allow_html=True)

    st.session_state.studentobservations = st.textarea(
        "Your observations about the image",
        value=st.session_state.studentobservations,
        height=100,
        placeholder="Describe colors, grain size, textures, layering, crystal shapes, etc.",
    )

    st.session_state.studentbestanswer = st.text_input(
        "Your best interpretation (name)",
        value=st.session_state.studentbestanswer,
        placeholder="e.g., well-sorted quartz sand, basalt, calcite crystal",
    )

    st.session_state.knownname = st.text_input(
        "Known name (instructor provided)",
        value=st.session_state.knownname,
        placeholder="What your instructor says this sample is",
    )

    st.markdown("---")

    start_disabled = st.session_state.imagedatauri is None
    if st.button(
        "Start first analysis",
        type="primary",
        disabled=start_disabled,
        use_container_width=True,
    ):
        try:
            start_first_analysis()
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.session_state.started and st.session_state.displaymessages:
        if st.button("Save conversation", use_container_width=True):
            try:
                result = save_conversation_to_file()
                if result:
                    filename, content = result
                    st.download_button(
                        label="Download conversation log",
                        data=content,
                        filename=filename,
                        mime="text/plain",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="gia-card">', unsafe_allow_html=True)
    st.markdown("<h3>Conversation</h3>", unsafe_allow_html=True)

    if not st.session_state.displaymessages:
        st.info("Upload an image and click **Start first analysis** to begin.")
    else:
        for msg in st.session_state.displaymessages:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a follow-up question or request a summary")
    if prompt:
        try:
            send_followup(prompt)
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)
