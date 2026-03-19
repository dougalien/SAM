import streamlit as st
from PIL import Image
import io
import os

from dotenv import load_dotenv
from openai import OpenAI

# ---- LOAD ENV & SET UP PERPLEXITY CLIENT ----
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY:
    raise RuntimeError("PERPLEXITY_API_KEY not found in environment variables.")

client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai",
)

# ---- STREAMLIT CONFIG ----
st.set_page_config(
    page_title="SAM – Ski Analysis Machine",
    page_icon="⛷️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- TITLE / HEADER ----
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.2rem;'>SAM</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size:0.9rem;'>Ski Analysis Machine for Instructors</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown(
    "Use SAM to reflect on a single image from your lesson. "
    "Upload a photo, describe the task, choose the level, and get a concise, supportive prescription for change."
)

# ---- INPUT FORM ----
with st.form("sam_form"):
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
        "Briefly describe what you see in the image",
        placeholder=(
            "Example: Skier in a wedge, skis forming a V, hips slightly behind feet, "
            "upper body facing downhill, inside ski looks more weighted."
        ),
        help="Describe stance, balance, turn shape, and anything notable in the image.",
    )

    image_file = st.file_uploader(
        "Lesson image (side or slight front/diagonal view works best)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a clear image with the whole skier visible if possible.",
    )

    focus = st.multiselect(
        "Optional focus areas",
        ["Stance & Balance", "Turn Shape", "Edge Control", "Pressure/Tilt", "Upper-Lower Body Separation"],
        help="Choose 1–2 focus areas if you’d like SAM to prioritize specific fundamentals.",
    )

    col_submit, col_reset = st.columns(2)
    with col_submit:
        submit = st.form_submit_button("Analyze with SAM ⛷️", use_container_width=True)
    with col_reset:
        reset = st.form_submit_button("Clear", use_container_width=True)

if reset:
    st.experimental_rerun()

# ---- MAIN ANALYSIS LOGIC ----
if submit:
    if image_file is None:
        st.warning("Please upload an image to analyze.")
        st.stop()

    if not scenario.strip():
        st.warning("Please describe the task or situation so SAM can respond in context.")
        st.stop()

    if not description.strip():
        st.warning("Please add a brief description of what you see in the image.")
        st.stop()

    st.subheader("Instructor view")
    st.image(image_file, width="stretch")

    with st.spinner("SAM is reviewing your description, stance, and fundamentals…"):
        system_prompt = """
You are SAM (Ski Analysis Machine), an AI assistant helping PSIA-aligned ski and snowboard instructors
reflect on lesson images and their own observations.

Guidelines:
- Audience: professional or aspiring instructors, not guests.
- Tone: clear, supportive, concise, and practical. Assume the instructor is on snow and has limited time.
- Focus on movement analysis: describe cause-and-effect between body movements and ski/snow performance.
- Base feedback on PSIA-style fundamentals: balanced, athletic stance; functional flex at ankles/knees/hips;
  primarily outside-ski balance in the shaping phase of the turn; progressive edging and pressure; upper/lower
  body working in harmony.
- Use guest-centered, positive language. Normalize errors, and offer 1–3 actionable focus points instead of a long list.
- Avoid medical advice, gear sales, and anything beyond technique, tactics, and teaching.

Output structure (markdown):
1. Very short summary (1–2 sentences) of what you infer from the instructor’s description.
2. "What’s working well" – 2–4 bullet points.
3. "Key opportunity" – 1–2 bullets with specific cause-and-effect.
4. "Prescription for change" – 2–4 bullets with concrete cues or simple drills the instructor can run next.

Adapt to the stated level:
- Beginner: emphasize basic stance, staying centered, simple turn shape, speed control, safety.
- Intermediate: refine edge control, consistent outside-ski balance, smoother transitions, turn shape.
- Advanced: refine edge angles, inside-half activity, timing of pressure and release, subtle tactical choices.

When you are unsure about an element in the description, use conditional language like "It appears that…" rather than stating it as a fact.
"""

        user_prompt = f"""
INSTRUCTOR CONTEXT
Ski school: Black Mountain, Jackson, NH.
Student level: {level}

Task / situation (instructor description):
{scenario}

Instructor's description of what they see in the image:
{description}

Optional focus areas: {", ".join(focus) if focus else "None given"}

You are NOT seeing the raw image; you are reasoning only from the instructor's description of the image and the task.
Provide specific, PSIA-style movement analysis and prescriptions for change based solely on this information.
"""

        try:
            response = client.chat.completions.create(
                model="sonar-pro",  # Perplexity Sonar chat model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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

    st.subheader("SAM’s analysis")
    st.markdown(analysis_markdown)

    st.caption(
        "SAM is a teaching aid for instructors and does not replace on-snow professional judgment."
    )
st.divider()
st.subheader("Chat with SAM about this lesson")

# Initialize chat history once
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m SAM. Ask me follow-up questions about this lesson, "
                "progressions, or alternative drills. I’ll answer based on general "
                "ski teaching best practices, not on the exact photo."
            ),
        }
    ]

# Show previous chat messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_chat = st.chat_input("Ask SAM a question about this student or lesson")

if user_chat:
    # Show user message
    st.session_state.chat_messages.append({"role": "user", "content": user_chat})
    with st.chat_message("user"):
        st.markdown(user_chat)

    # Build conversation for the API
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
- If you lack context (no image, limited description), ask 1 clarifying question before giving detailed advice.
"""

    chat_messages_for_api = [
        {"role": "system", "content": system_prompt_chat},
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
                    model="sonar-pro",
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
