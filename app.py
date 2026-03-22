import os
import io
import base64
import mimetypes
from PIL import Image

import streamlit as st
from openai import OpenAI

# ---- SET UP PERPLEXITY CLIENT FROM SECRETS ----
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai",
)

MODEL = "sonar-pro"  # Same base model as GIA

# ---- STREAMLIT CONFIG ----
st.set_page_config(
    page_title="SAM – Ski Analysis Machine",
    page_icon="⛷️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- SKI-SCHOOL THEME (CSS) ----
st.markdown(
    """
    <style>
    :root {
        --sam-bg: #F4F8FB;          /* soft snow/sky */
        --sam-primary: #0D47A1;     /* deep alpine blue */
        --sam-secondary: #FFB74D;   /* warm lodge orange */
        --sam-accent: #1E88E5;      /* bright blue accent */
        --sam-text: #1A2733;        /* dark slate */
        --sam-muted: #5C6B73;       /* muted gray-blue */
    }

    .main {
        background: radial-gradient(circle at top, #E3F2FD 0, #F4F8FB 45%, #FFFFFF 100%);
    }

    .sam-header {
        text-align: center;
        padding-top: 0.4rem;
        padding-bottom: 0.1rem;
        color: var(--sam-primary);
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    .sam-subtitle {
        text-align: center;
        color: var(--sam-muted);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .sam-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background-color: rgba(13, 71, 161, 0.08);
        color: var(--sam-accent);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .sam-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(13, 71, 161, 0.08);
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
    }

    .sam-card h3 {
        margin-top: 0;
        color: var(--sam-primary);
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    .sam-body-text {
        color: var(--sam-text);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .sam-helper-text {
        color: var(--sam-muted);
        font-size: 0.85rem;
    }

    .sam-separator {
        text-align: center;
        color: var(--sam-secondary);
        font-size: 1.2rem;
        margin: 0.6rem 0 1.0rem 0;
    }

    /* Primary buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, var(--sam-primary), var(--sam-accent));
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        font-size: 0.98rem;
        font-weight: 600;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.18);
        cursor: pointer;
        transition: transform 0.06s ease-in-out, box-shadow 0.06s ease-in-out, filter 0.06s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    }

    div.stButton > button:first-child:active {
        transform: translateY(0px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.20);
    }

    /* Secondary buttons (Clear) */
    .sam-clear button {
        background: transparent !important;
        color: var(--sam-primary) !important;
        border: 1px solid rgba(13, 71, 161, 0.4) !important;
        box-shadow: none !important;
    }

    .sam-clear button:hover {
        background: rgba(13, 71, 161, 0.06) !important;
    }

    /* Chat message styling tweaks */
    .stChatMessage .markdown-text-container p {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- UTIL: FILE TO DATA URI ----
def file_to_data_uri(uploaded_file):
    """Convert a Streamlit UploadedFile to (data_uri, mime, name)."""
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type
    if not mime:
        mime = mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return data_uri, mime, uploaded_file.name

# ---- HEADER ----
st.markdown(
    """
    <h1 class="sam-header">SAM – Ski Analysis Machine ⛷️</h1>
    <div class="sam-subtitle">
        <span class="sam-badge">Instructor Tool</span>
        <br/><br/>
        PSIA-aligned movement analysis helper for ski and snowboard instructors.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sam-separator">🎿 ❄️ 🎿</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="sam-body-text">'
    "Use SAM to reflect on one or two images from your lesson. Upload photo(s), describe the task, "
    "choose the level, and get a concise, supportive prescription for change."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("")  # small spacer

# ---- INPUT FORM ----
with st.form("sam_form"):
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="sam-card">', unsafe_allow_html=True)
        st.markdown("<h3>Lesson context</h3>", unsafe_allow_html=True)

        level = st.selectbox(
            "Student level",
            ["Beginner", "Intermediate", "Advanced"],
            help="Roughly match PSIA zones: beginner/novice, intermediate, advanced.",
        )

        scenario = st.text_area(
            "Describe the task / situation",
            placeholder=(
                "Example: Wedge turn to the left on green terrain, mid-turn. "
                "Student tends to lean back and stem the uphill ski at initiation."
            ),
            help="Mention terrain, phase of turn, typical issues you’ve noticed, and the main goal.",
        )

        description = st.text_area(
            "Briefly describe what you see in the image(s)",
            placeholder=(
                "Example: Skier in a wedge, skis forming a V, hips slightly behind feet, "
                "upper body facing downhill, inside ski looks more weighted."
            ),
            help="Describe stance, balance, turn shape, and anything notable in the image(s).",
        )

        focus = st.multiselect(
            "Optional focus areas",
            [
                "Stance & Balance",
                "Turn Shape",
                "Edge Control",
                "Pressure/Tilt",
                "Upper-Lower Body Separation",
            ],
            help="Choose 1–2 focus areas if you’d like SAM to prioritize specific fundamentals.",
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="sam-card">', unsafe_allow_html=True)
        st.markdown("<h3>Lesson images</h3>", unsafe_allow_html=True)

        image_file_1 = st.file_uploader(
            "Lesson image 1 (main view)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Use a clear image with the whole skier visible if possible.",
            key="image1",
        )
        image_file_2 = st.file_uploader(
            "Lesson image 2 (optional alternate view)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Optional: second angle or later frame in the turn.",
            key="image2",
        )

        st.markdown(
            '<p class="sam-helper-text">'
            "Tip: A clear side or ¾ view mid-turn usually gives the best information."
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    col_submit, col_reset = st.columns([2, 1])
    with col_submit:
        submit = st.form_submit_button("Analyze with SAM ⛷️", use_container_width=True)
    with col_reset:
        with st.container():
            st.markdown('<div class="sam-clear">', unsafe_allow_html=True)
            reset = st.form_submit_button("Clear", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if reset:
        st.experimental_rerun()

analysis_markdown = None

# ---- MAIN ANALYSIS LOGIC ----
if submit:
    if image_file_1 is None and image_file_2 is None:
        st.warning("Please upload at least one image to analyze.")
        st.stop()

    if not scenario.strip():
        st.warning("Please describe the task or situation so SAM can respond in context.")
        st.stop()

    if not description.strip():
        st.warning("Please add a brief description of what you see in the image(s).")
        st.stop()

    st.markdown('<div class="sam-card">', unsafe_allow_html=True)
    st.markdown("<h3>Instructor view</h3>", unsafe_allow_html=True)

    cols = st.columns(2)
    if image_file_1 is not None:
        with cols[0]:
            st.image(image_file_1, caption="Image 1", use_container_width=True)
    if image_file_2 is not None:
        with cols[1]:
            st.image(image_file_2, caption="Image 2", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("SAM is reviewing your images, description, stance, and fundamentals…"):
        system_prompt = """
        You are SAM (Ski Analysis Machine), an AI assistant helping PSIA-aligned ski and snowboard instructors reflect on lesson images and their own observations.

        Guidelines:
        - Audience: professional or aspiring instructors, not guests.
        - Tone: clear, supportive, concise, and practical. Assume the instructor is on snow and has limited time.
        - Focus on movement analysis: describe cause-and-effect between body movements and ski/snow performance.
        - Base feedback on PSIA-style fundamentals: balanced, athletic stance; functional flex at ankles/knees/hips; primarily outside-ski balance in the shaping phase of the turn; progressive edging and pressure; upper/lower body working in harmony.
        - Use guest-centered, positive language. Normalize errors, and offer 1–3 actionable focus points instead of a long list.
        - Avoid medical advice, gear sales, and anything beyond technique, tactics, and teaching.

        You are seeing one or two lesson images plus the instructor’s text descriptions. Use the images to anchor your movement analysis, and the text to understand context and focus.

        Output structure (markdown):
        1. Very short summary (1–2 sentences) of what you infer from the images and instructor’s description.
        2. "What’s working well" – 2–4 bullet points.
        3. "Key opportunity" – 1–2 bullets with specific cause-and-effect.
        4. "Prescription for change" – 2–4 bullets with concrete cues or simple drills the instructor can run next.

        Adapt to the stated level:
        - Beginner: emphasize basic stance, staying centered, simple turn shape, speed control, safety.
        - Intermediate: refine edge control, consistent outside-ski balance, smoother transitions, turn shape.
        - Advanced: refine edge angles, inside-half activity, timing of pressure and release, subtle tactical choices.

        When you are unsure about an element, use conditional language like "It appears that…" rather than stating it as a fact.
        """

        user_text = f"""
        INSTRUCTOR CONTEXT
        Ski school: Black Mountain, Jackson, NH.
        Student level: {level}
        Task / situation (instructor description): {scenario}
        Instructor's description of what they see in the image(s): {description}
        Optional focus areas: {", ".join(focus) if focus else "None given"}.

        You are seeing the uploaded lesson photo(s) plus this description.
        Provide specific, PSIA-style movement analysis and prescriptions for change based on both the images and the text.
        """

        # Build messages with text + image_url blocks
        user_content = [
            {"type": "text", "text": user_text.strip()},
        ]

        if image_file_1 is not None:
            data_uri_1, _, _ = file_to_data_uri(image_file_1)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri_1},
                }
            )

        if image_file_2 is not None:
            data_uri_2, _, _ = file_to_data_uri(image_file_2)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri_2},
                }
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=800,
            )
            if not response.choices:
                st.error("Perplexity API returned no choices. Please try again.")
                st.stop()
            analysis_markdown = response.choices[0].message.content
        except Exception as e:
            st.error("Error from SAM’s AI engine. Check your API key, model name, and logs.")
            st.caption(f"Debug info: {e}")
            st.stop()

    if analysis_markdown:
        st.markdown('<div class="sam-card">', unsafe_allow_html=True)
        st.markdown("<h3>SAM’s analysis</h3>", unsafe_allow_html=True)
        st.markdown(analysis_markdown)
        st.caption(
            "SAM is a teaching aid for instructors and does not replace on-snow professional judgment."
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
st.divider()

# ---- FOLLOW-UP CHAT (TEXT ONLY) ----
st.subheader("Chat with SAM about this lesson")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m SAM. Ask me follow-up questions about this lesson, "
                "progressions, or alternative drills. I’ll answer based on general "
                "ski teaching best practices and the context you’ve given."
            ),
        }
    ]

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_chat = st.chat_input("Ask SAM a question about this student or lesson")

if user_chat:
    st.session_state.chat_messages.append({"role": "user", "content": user_chat})
    with st.chat_message("user"):
        st.markdown(user_chat)

    system_prompt_chat = """
    You are SAM (Ski Analysis Machine), an AI assistant helping PSIA-aligned snowsports instructors.
    You are in a follow-up chat after an initial movement analysis.

    Guidelines:
    - Audience: instructors, not guests.
    - Tone: supportive, practical, concise.
    - Stay focused on ski technique, tactics, and teaching progressions.
    - Use PSIA-style fundamentals and language where appropriate.
    - Offer 1–3 clear options or drills rather than long lists.
    - If asked about safety or terrain choice, give conservative, general guidance.
    - If you lack context, ask 1 clarifying question before giving detailed advice.
    """

    chat_messages_for_api = [
        {"role": "system", "content": system_prompt_chat.strip()},
        {
            "role": "user",
            "content": (
                f"Instructor at Black Mountain, Jackson NH. "
                f"Student level: {level if 'level' in locals() else 'Unknown'}. "
                f"Original task/situation: {scenario if 'scenario' in locals() else 'Not provided'}. "
                f"Optional focus areas: {', '.join(focus) if 'focus' in locals() and focus else 'None given'}.\n\n"
                f"Follow-up question from the instructor:\n{user_chat}"
            ),
        },
    ]

    with st.chat_message("assistant"):
        with st.spinner("SAM is thinking…"):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=chat_messages_for_api,
                    temperature=0.5,
                    max_tokens=700,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = (
                    "I ran into an error contacting the AI engine. "
                    "Please check your API settings and try again.\n\n"
                    f"(Debug info: {e})"
                )
            st.markdown(reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
