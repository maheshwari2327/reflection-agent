import functools
import os
import re
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from tools import count_words

# --- System Prompts ---

WRITER_PROMPT = """You are a skilled writer. Write a short, clear piece (max 100 words)
for the user's request. Return ONLY the text itself — no title, no intro like "Here is..."."""

CRITIC_PROMPT = """You are a strict but fair editor. Review the draft against the request.
Check: clarity, tone, grammar, and whether it fully answers the request.
HARD RULE: the draft must be 100 words or fewer. If it is longer, it is NOT approved —
tell the writer to cut it.
- If it is good enough AND within the limit, reply with exactly: APPROVED
- Otherwise reply with 2-3 short, specific suggestions (one line each)."""

REVISER_PROMPT = """You are a writer revising your draft.
Rewrite it applying the editor's feedback, but keep it to 100 words or fewer
(cut words if needed — never make it longer).
Return ONLY the improved text. Do NOT start with "Here is the revised text" or any intro."""

# --- Configuration & State ---

MAX_ROUNDS = 3

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Default LLM
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.5)


class AgentState(TypedDict):
    request: str
    draft: str
    feedback: str
    rounds: int


# --- Helper Functions ---


def clean(text: str) -> str:
    """Strip 'Here is the revised text:' and similar intro lines from model output."""
    if not text:
        return ""
    text = text.strip()

    # Pattern for common intro phrases like "Here is the revised text:", "Here's the draft:", etc.
    intro_prefix_regex = re.compile(
        r"^(?:(?:certainly|sure|of course)[!.,]?\s*)?"
        r"(?:(?:here\s+(?:is|are)|here\'s)\s+(?:the\s+)?(?:short\s+piece|revised\s+(?:text|draft|piece)|improved\s+(?:text|draft)|draft|email|response|text)?[^:\n]*:|"
        r"revised\s+(?:text|draft)\s*:|"
        r"draft\s*:)",
        re.IGNORECASE,
    )

    match = intro_prefix_regex.match(text)
    if match:
        text = text[match.end():].strip()

    # Line by line check for standalone intro lines
    lines = text.splitlines()
    while lines:
        first_line = lines[0].strip()
        lower = first_line.lower()
        if (
            (lower.startswith("here is") or lower.startswith("here's"))
            and (first_line.endswith(":") or len(first_line.split()) <= 6)
        ) or (
            lower in ("revised text:", "revised draft:", "draft:", "here is the revised text:", "here is the draft:")
        ):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    cleaned = "\n".join(lines).strip()

    # Strip surrounding quotes if the model wrapped the output in quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    return cleaned


# --- Nodes ---


def write(state: AgentState, model: Optional[ChatOllama] = None) -> dict:
    """Writer node: generate the initial draft."""
    active_model = model or llm
    response = active_model.invoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=state["request"]),
        ]
    )
    draft = clean(response.content.strip())
    length_info = count_words.invoke(draft)
    print(f"\n  ✍️  [writer] first draft ({length_info})")
    return {"draft": draft, "rounds": 0}


def critique(state: AgentState, model: Optional[ChatOllama] = None) -> dict:
    """Critic node: review the draft and provide APPROVED or suggestions."""
    active_model = model or llm
    draft = state.get("draft", "")
    word_count = len(draft.split())

    critic_input = (
        f"User Request:\n{state['request']}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Word count: {word_count} (limit 100)"
    )

    response = active_model.invoke(
        [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=critic_input),
        ]
    )
    feedback = response.content.strip()

    # HARD RULE: override if draft > 100 words
    if word_count > 100 and feedback.startswith("APPROVED"):
        feedback = (
            f"Draft exceeds the 100-word limit ({word_count} words). "
            f"Please cut it down to 100 words or fewer."
        )

    print(f"  🧐 [critic] {feedback}")
    return {"feedback": feedback}


def revise(state: AgentState, model: Optional[ChatOllama] = None) -> dict:
    """Reviser node: revise the draft based on critic feedback."""
    active_model = model or llm
    rounds = state.get("rounds", 0) + 1

    reviser_input = (
        f"User Request:\n{state['request']}\n\n"
        f"Current Draft:\n{state['draft']}\n\n"
        f"Editor's Feedback:\n{state['feedback']}"
    )

    response = active_model.invoke(
        [
            SystemMessage(content=REVISER_PROMPT),
            HumanMessage(content=reviser_input),
        ]
    )
    revised_draft = clean(response.content.strip())
    length_info = count_words.invoke(revised_draft)
    print(f"  🔁 [reviser] revision #{rounds} ({length_info})")
    return {"draft": revised_draft, "rounds": rounds}


# --- Conditional Edge ---


def should_continue(state: AgentState) -> str:
    """Determine whether to stop (approved or max rounds) or revise."""
    feedback = state.get("feedback", "").strip()
    rounds = state.get("rounds", 0)

    # Stop if approved
    if feedback.startswith("APPROVED"):
        return END

    # Stop if max rounds reached
    if rounds >= MAX_ROUNDS:
        return END

    # Otherwise revise
    return "revise"


# --- Graph & Entry Point ---


def build_agent(custom_llm: Optional[ChatOllama] = None):
    """Build and compile the LangGraph reflection agent."""
    active_model = custom_llm if custom_llm is not None else llm

    workflow = StateGraph(AgentState)

    workflow.add_node("write", functools.partial(write, model=active_model))
    workflow.add_node("critique", functools.partial(critique, model=active_model))
    workflow.add_node("revise", functools.partial(revise, model=active_model))

    workflow.add_edge(START, "write")
    workflow.add_edge("write", "critique")
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revise": "revise",
            END: END,
        },
    )
    workflow.add_edge("revise", "critique")

    return workflow.compile()


def ask(request: str, agent=None) -> dict:
    """Run the reflection agent on a request."""
    if agent is None:
        agent = build_agent()

    initial_state: AgentState = {
        "request": request,
        "draft": "",
        "feedback": "",
        "rounds": 0,
    }

    return agent.invoke(initial_state)


def main():
    """Interactive CLI for the Reflection Agent."""
    print("=" * 60)
    print(" Reflection Agent CLI (LangChain + LangGraph + Ollama) ")
    print(" Type 'quit', 'exit', or 'q' to exit. ")
    print("=" * 60 + "\n")

    agent = build_agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Reject inputs shorter than 4 words
        words = user_input.split()
        if len(words) < 4:
            print("  ⚠️ Request must be at least 4 words long. Please describe what you want written.\n")
            continue

        try:
            result = ask(user_input, agent=agent)
            print(f"\nFinal: {result['draft']}\n")
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")


if __name__ == "__main__":
    main()
