import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from tools import count_words
from agent import (
    clean,
    should_continue,
    write,
    critique,
    revise,
    build_agent,
    ask,
    AgentState,
    MAX_ROUNDS,
    WRITER_PROMPT,
    CRITIC_PROMPT,
    REVISER_PROMPT,
)
from langgraph.graph import END


class TestTools(unittest.TestCase):
    def test_count_words_contract(self):
        """Verify the exact docstring contract."""
        self.assertEqual(
            count_words.description,
            "Count the words in a piece of text and report whether it is short, medium or long.",
        )

    def test_count_words_short(self):
        result = count_words.invoke("Hello world this is a test.")
        self.assertEqual(result, "6 words (short)")

    def test_count_words_medium(self):
        text = "word " * 81
        result = count_words.invoke(text)
        self.assertEqual(result, "81 words (medium)")

    def test_count_words_long(self):
        text = "word " * 95
        result = count_words.invoke(text)
        self.assertEqual(result, "95 words (long)")


class TestClean(unittest.TestCase):
    def test_clean_various_intros(self):
        cases = [
            ("Here is the revised text:\n\nDear Boss, please approve.", "Dear Boss, please approve."),
            ("Here is the revised text: Dear Boss, please approve.", "Dear Boss, please approve."),
            ("Here's the draft:\nDear Boss, please approve.", "Dear Boss, please approve."),
            ("Certainly! Here is the revised text:\nDear Boss, please approve.", "Dear Boss, please approve."),
            ("Dear Boss, please approve.", "Dear Boss, please approve."),
            ('"Dear Boss, please approve."', "Dear Boss, please approve."),
        ]
        for inp, expected in cases:
            self.assertEqual(clean(inp), expected)


class TestShouldContinue(unittest.TestCase):
    def test_approved(self):
        state: AgentState = {
            "request": "test request here",
            "draft": "draft",
            "feedback": "APPROVED",
            "rounds": 0,
        }
        self.assertEqual(should_continue(state), END)

    def test_approved_with_text(self):
        state: AgentState = {
            "request": "test request here",
            "draft": "draft",
            "feedback": "APPROVED - Great draft!",
            "rounds": 1,
        }
        self.assertEqual(should_continue(state), END)

    def test_max_rounds(self):
        state: AgentState = {
            "request": "test request here",
            "draft": "draft",
            "feedback": "Needs work.",
            "rounds": MAX_ROUNDS,
        }
        self.assertEqual(should_continue(state), END)

    def test_revise_loop(self):
        state: AgentState = {
            "request": "test request here",
            "draft": "draft",
            "feedback": "1. Make it shorter.\n2. Fix tone.",
            "rounds": 1,
        }
        self.assertEqual(should_continue(state), "revise")


class TestNodes(unittest.TestCase):
    def test_write_node(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Here is the draft:\nDear Boss, I need Friday off.")

        state: AgentState = {
            "request": "Write email asking for Friday off",
            "draft": "",
            "feedback": "",
            "rounds": 0,
        }

        res = write(state, model=mock_llm)
        self.assertEqual(res["draft"], "Dear Boss, I need Friday off.")
        self.assertEqual(res["rounds"], 0)

    def test_critique_approved(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="APPROVED")

        state: AgentState = {
            "request": "Write email asking for Friday off",
            "draft": "Dear Boss, I need Friday off.",
            "feedback": "",
            "rounds": 0,
        }

        res = critique(state, model=mock_llm)
        self.assertEqual(res["feedback"], "APPROVED")

    def test_critique_over_100_words_override(self):
        """Code must override APPROVED if draft is over 100 words."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="APPROVED")

        long_draft = "word " * 105
        state: AgentState = {
            "request": "Write email asking for Friday off",
            "draft": long_draft,
            "feedback": "",
            "rounds": 0,
        }

        res = critique(state, model=mock_llm)
        self.assertFalse(res["feedback"].startswith("APPROVED"))
        self.assertIn("100-word limit", res["feedback"])

    def test_revise_node(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Here is the revised text:\nDear Boss, I am taking Friday off.")

        state: AgentState = {
            "request": "Write email asking for Friday off",
            "draft": "Old draft",
            "feedback": "Make it more concise.",
            "rounds": 1,
        }

        res = revise(state, model=mock_llm)
        self.assertEqual(res["draft"], "Dear Boss, I am taking Friday off.")
        self.assertEqual(res["rounds"], 2)


class TestFullAgentGraph(unittest.TestCase):
    def test_graph_approval_after_one_revision(self):
        # Call 1: write -> initial draft
        # Call 2: critique -> feedback (needs edit)
        # Call 3: revise -> revised draft
        # Call 4: critique -> APPROVED
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="First draft of the letter."),
            AIMessage(content="1. Make it more professional.\n2. Add closing."),
            AIMessage(content="Dear Boss,\nI request Friday off.\nBest regards."),
            AIMessage(content="APPROVED"),
        ]

        agent = build_agent(custom_llm=mock_llm)
        result = ask("Please write a formal email", agent=agent)

        self.assertEqual(result["rounds"], 1)
        self.assertTrue(result["feedback"].startswith("APPROVED"))
        self.assertIn("Best regards.", result["draft"])

    def test_graph_immediate_approval(self):
        # Call 1: write -> initial draft
        # Call 2: critique -> APPROVED
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Dear Boss,\nRequesting Friday off. Thank you."),
            AIMessage(content="APPROVED"),
        ]

        agent = build_agent(custom_llm=mock_llm)
        result = ask("Please write a formal email", agent=agent)

        self.assertEqual(result["rounds"], 0)
        self.assertTrue(result["feedback"].startswith("APPROVED"))


if __name__ == "__main__":
    unittest.main()
