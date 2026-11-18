import os
import requests
import streamlit as st
from langchain_ollama import ChatOllama

from agent import AgentState, build_agent
from tools import count_words

# --- Page Configuration ---
st.set_page_config(
    page_title="Reflection Agent",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .step-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a90e2;
        background-color: rgba(74, 144, 226, 0.05);
        margin-bottom: 1rem;
    }
    .critic-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        background-color: rgba(243, 156, 18, 0.05);
        margin-bottom: 1rem;
    }
    .reviser-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9b59b6;
        background-color: rgba(155, 89, 182, 0.05);
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 12px;
        background: #eef2f7;
        color: #2c3e50;
        margin-left: 0.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    st.markdown("Configure the local LLM and execution loop.")

    model_name = st.text_input("Ollama Model", value="llama3.2")
    base_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # Health check for Ollama
    if st.button("🔌 Check Ollama Connection"):
        try:
            res = requests.get(base_url, timeout=3)
            if res.status_code == 200:
                st.success("✅ Ollama is running and accessible!")
            else:
                st.warning(f"Ollama returned status {res.status_code}")
        except Exception as e:
            st.error(f"❌ Could not connect to Ollama: {e}\nMake sure `ollama serve` is running.")

    st.divider()
    st.markdown("### 🔄 Reflection Pattern")
    st.caption(
        """
        1. **Writer**: Writes initial draft (max 100 words).
        2. **Critic**: Evaluates clarity, grammar, and enforces the 100-word limit.
        3. **Reviser**: Improves the draft if not approved (up to 3 rounds).
        """
    )


# --- Main Application ---
st.markdown('<div class="main-header">🔄 Reflection Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangChain + LangGraph iterative writing assistant powered by Ollama</div>',
    unsafe_allow_html=True,
)

# Quick Suggestion Chips
st.markdown("**Try an example request:**")
col1, col2, col3 = st.columns(3)
if col1.button("📧 Email for Friday off", use_container_width=True):
    st.session_state["prompt_input"] = "Write a short email asking my boss for Friday off"
if col2.button("📢 Team lunch announcement", use_container_width=True):
    st.session_state["prompt_input"] = "Write an announcement to the team about lunch this Friday"
if col3.button("🚀 Product launch post", use_container_width=True):
    st.session_state["prompt_input"] = "Write a short post announcing the launch of our new reflection agent"

# User input text area
user_request = st.text_area(
    "What would you like the agent to write?",
    value=st.session_state.get("prompt_input", ""),
    height=100,
    placeholder="e.g. Write a short email asking my boss for Friday off",
)

col_btn, col_clear = st.columns([1, 5])
start_run = col_btn.button("✨ Generate & Refine", type="primary", use_container_width=True)

if start_run:
    words = user_request.strip().split()
    if len(words) < 4:
        st.warning("⚠️ Request must be at least 4 words long. Please describe what you want written.")
    else:
        # Initialize LLM with sidebar configuration
        try:
            llm = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
            agent = build_agent(custom_llm=llm)
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {e}")
            st.stop()

        status_container = st.status("🚀 Running Reflection Loop...", expanded=True)
        final_draft = ""
        final_feedback = ""
        rounds_taken = 0

        try:
            initial_state: AgentState = {
                "request": user_request,
                "draft": "",
                "feedback": "",
                "rounds": 0,
            }

            for step in agent.stream(initial_state):
                for node_name, node_output in step.items():
                    if node_name == "write":
                        draft = node_output.get("draft", "")
                        final_draft = draft
                        stats = count_words.invoke(draft)
                        with status_container:
                            st.markdown(
                                f"""
                                <div class="step-card">
                                    <strong>✍️ [Writer] First Draft</strong> <span class="badge">{stats}</span>
                                    <p style="margin-top: 0.5rem; white-space: pre-wrap;">{draft}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    elif node_name == "critique":
                        feedback = node_output.get("feedback", "")
                        final_feedback = feedback
                        with status_container:
                            if feedback.startswith("APPROVED"):
                                st.markdown(
                                    f"""
                                    <div class="critic-card" style="border-left-color: #27ae60; background-color: rgba(39, 174, 96, 0.05);">
                                        <strong style="color: #27ae60;">🧐 [Critic] {feedback}</strong>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class="critic-card">
                                        <strong>🧐 [Critic] Suggestions:</strong>
                                        <p style="margin-top: 0.5rem; white-space: pre-wrap;">{feedback}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    elif node_name == "revise":
                        draft = node_output.get("draft", "")
                        rounds = node_output.get("rounds", 1)
                        rounds_taken = rounds
                        final_draft = draft
                        stats = count_words.invoke(draft)
                        with status_container:
                            st.markdown(
                                f"""
                                <div class="reviser-card">
                                    <strong>🔁 [Reviser] Revision #{rounds}</strong> <span class="badge">{stats}</span>
                                    <p style="margin-top: 0.5rem; white-space: pre-wrap;">{draft}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            status_container.update(label="✅ Reflection Complete!", state="complete", expanded=False)

            # Display final approved output
            st.success("🎉 Final Approved Result")
            st.text_area("Final Draft", value=final_draft, height=140)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Words", len(final_draft.split()))
            m_col2.metric("Revision Rounds", rounds_taken)
            m_col3.metric("Status", "APPROVED" if final_feedback.startswith("APPROVED") else "Max Rounds Reached")

        except Exception as e:
            status_container.update(label="❌ Failed", state="error")
            st.error(f"An error occurred while running the reflection agent: {e}")
            st.info("💡 Hint: Ensure Ollama is running (`ollama serve`) and the model is downloaded (`ollama pull llama3.2`).")
