"""AMA KBQA Assistant - demo entry point.

This is a single-page demo: the only page is the Chat interface, served at the
root URL. The multi-page dashboard (Evaluation, Settings, ...) is intentionally
stripped out for the public demo.

On first visit an "About" dialog explains the project and how the agent works.
It shows once per session; a small floating "?" button reopens it anytime.
"""

import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css

st.set_page_config(
    page_title="AMA KBQA Assistant",
    page_icon="✨",
    layout="centered",
)

inject_css()

ABOUT_MD = """
### AMA KBQA: Ask Me Anything about Knowledge Graphs

This is a research demo of a **Knowledge Base Question Answering** system. It
answers natural-language questions by *grounding* every answer in a structured
knowledge graph, instead of relying on whatever a language model happens to
remember.

#### How the agent thinks

Rather than answering in one shot, the agent works like a researcher:

1. **Understand** the question and pick out the entities and relations it mentions.
2. **Search** the knowledge graph. It looks entities and relations up in a vector
   database (Qdrant) and runs precise SPARQL queries against the graph (Virtuoso).
3. **Reason in a loop**, calling tools, reading the results, and deciding what to
   look up next, until it has gathered enough evidence.
4. **Answer from evidence.** If the graph cannot support an answer, the agent says
   so rather than guessing.

#### The agents

- **Orchestrator** routes your question to the right specialist automatically.
- **KQAPro** answers factual questions (films, geography, science facts).
- **SciQA** answers scientific-research questions over the Open Research Knowledge
  Graph (ORKG).

Pick a specific agent (KQAPro or SciQA) to have a **follow-up conversation** and
drill down on an answer.

#### What you can see

Every answer shows its reasoning trace, how long it took, how many tokens it used,
and an *illustrative* cost. The cost is for intuition only: this demo runs on a
free KIT-hosted endpoint, so nobody is actually billed.

:gray[Click the **?** button in the corner anytime to reopen this panel.]
"""


@st.dialog("About this demo", width="large")
def _about_dialog() -> None:
    st.markdown(ABOUT_MD)


# Show the About dialog once per browser session (on first load).
if not st.session_state.get("_about_seen"):
    st.session_state["_about_seen"] = True
    _about_dialog()

# Small floating "?" button to reopen the About dialog. Positioned in the corner
# via the `.st-key-about_help_btn` CSS rule in styling.py.
if st.button("?", key="about_help_btn", help="About this demo"):
    _about_dialog()

# Single-page navigation: Chat only, served as the default (root) page. Using
# st.navigation disables Streamlit's automatic pages/ discovery, and
# position="hidden" removes the page switcher entirely.
chat_page = st.Page("chat.py", title="Chat", icon=":material/chat:", default=True)
st.navigation([chat_page], position="hidden").run()
