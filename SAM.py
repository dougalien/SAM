import streamlit as st
from PIL import Image
import io
import base64

# ---- CONFIG ----
st.set_page_config(
    page_title="SAM – Ski Analysis Machine",
    page_icon="⛷️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- HEADER ----
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

# ---- INPUTS ----
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

# ---- HELPER: ENCODE IMAGE ----
def encode_image_to_base64(file) -> str:
    if file is None:
        return ""
    image = Image.open(file)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64

# ---- CORE ANALYSIS ----
if submit:
    if image_file is None:
        st.warning("Please upload an image to analyze.")
        st.stop()

    if not scenario.strip():
        st.warning("Please describe the task or situation so SAM can respond in context.")
        st.stop()

    st.subheader("Instructor view")
    st.image(image_file, use_column_width=True)

    with st.spinner("SAM is reviewing the stance, turn phase, and fundamentals…"):
        img_b64 = encode_image_to_base64(image_file)

        # SYSTEM PROMPT FOR PSAI / LLM
        system_prompt = f"""
You are SAM (Ski Analysis Machine), an AI assistant helping PSIA-aligned ski and snowboard instructors
reflect on still images from their lessons.

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
1. Very short summary (1–2 sentences) of what you see.
2. "What’s working well" – 2–4 bullet points.
3. "Key opportunity" – 1–2 bullets with specific cause-and-effect.
4. "Prescription for change" – 2–4 bullets with concrete cues or simple drills the instructor can run next.

Adapt to the stated level:
- Beginner: emphasize basic stance, staying centered, simple turn shape, speed control, safety.
- Intermediate: refine edge control, consistent outside-ski balance, smoother transitions, turn shape.
- Advanced: refine edge angles, inside-half activity, timing of pressure and release, subtle tactical choices.

When you are unsure about an element in the image, use conditional language like "It appears that…" rather than stating it as a fact.
"""

        # USER PROMPT WITH CONTEXT
        user_prompt = f"""
INSTRUCTOR CONTEXT
Ski school: Black Mountain, Jackson, NH.
Student level: {level}
Scenario (instructor description): {scenario}
Optional focus areas: {", ".join(focus) if focus else "None given"}

IMAGE
You receive a single still image encoded in base64 PNG format.
Only infer what is reasonably visible in the still image; do not invent terrain or conditions.
Base64 image data:
{img_b64[:2000]}  # truncated for token safety; you will receive the full data in actual calls
"""

        # TODO: Call your PSAI / LLM client here, passing system_prompt, user_prompt, and the full base64 image.
        # For example, with OpenAI-style client (pseudo-code):
        #
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4.1-mini",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     temperature=0.4,
        # )
        # analysis_markdown = response.choices[0].message.content
        #
        # For now, we’ll just stub a placeholder:

        analysis_markdown = """
**Summary**  
You’ve captured a useful moment in the turn where stance and outside-ski balance can be refined.

**What’s working well**
- Skier shows an overall athletic stance with some flex in the ankles and knees.
- Upper body is generally facing down the hill rather than excessively rotating.
- Both skis are engaged with the snow, providing a good platform for further improvement.

**Key opportunity**
- It appears the skier’s hips are drifting slightly behind the feet, which reduces pressure on the front of the boots and makes turn shaping less precise.
- The inside ski looks more weighted than ideal, which can limit edge grip and control from the outside ski.

**Prescription for change**
- Ask the student to "feel both shins gently against the front of the boots" throughout the turn to bring the hips more over the feet.
- Use a simple traverse-to-garland drill, focusing on balancing over the outside ski and keeping the inside ski light but guided alongside.
- On mellow terrain, have them ski a series of medium-radius turns with the cue "outside ski strong, inside ski along for the ride," checking that they feel more pressure under the outside foot.
"""

    st.subheader("SAM’s analysis")
    st.markdown(analysis_markdown)

    st.caption(
        "SAM is a teaching aid for instructors and does not replace on-snow professional judgment."
    )
