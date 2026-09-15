import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import { IconClose } from "./icons";

// ABOUT_MD from ama_kbqa/frontend/app.py, adapted for this page (the gray
// Streamlit markup on the last line becomes a muted paragraph).
const ABOUT_MD = `
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

- **Orchestrator (Router)** routes your question to the single best specialist
  automatically. This is the dispatch strategy evaluated in the paper.
- **Orchestrator (Federated)** *(experimental)* may ask both specialists at
  once and combine their answers with an extra fusion step. It can catch
  questions that span both knowledge graphs, but is slower and uses about
  twice the tokens; its numbers are not the paper's.
- **KQAPro** answers factual questions (films, geography, science facts).
- **SciQA** answers scientific-research questions over the Open Research Knowledge
  Graph (ORKG).

Pick a specific agent (KQAPro or SciQA) to have a **follow-up conversation** and
drill down on an answer.

#### What you can see

Every answer shows its reasoning trace, how long it took, how many tokens it used,
and an *illustrative* cost. The cost is for intuition only: this demo runs on a
free KIT-hosted endpoint, so nobody is actually billed.

#### What can go wrong

This is a research prototype, so expect a few rough edges:

- **Entity linking** can resolve a name to the wrong entity (e.g. a person with
  a similar name or occupation), producing a confidently wrong answer. Check
  the linked entity in the reasoning trace if something looks off.
- **Definitional questions** ("What is X?") often fail when the knowledge graph
  has no explicit definition. The system only reports what is in the graph,
  not what it separately "knows".
- **"I don't know"** is a valid answer: the agent says so whenever it cannot
  ground a claim in the graph, even if the fact seems like common knowledge.
- **Broad or superlative questions** ("the most famous...") can be slow and
  sometimes return no results, since they require expensive aggregation over
  the graph.
- **Answers take time**, typically one to a few minutes, depending on question
  complexity and model load.
- **Follow-ups aren't remembered yet.** Each question is answered
  independently; corrections and clarifications don't carry over.
`;

interface AboutDialogProps {
  open: boolean;
  onClose: () => void;
}

export function AboutDialog({ open, onClose }: AboutDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby="about-title"
      onClose={onClose}
      onClick={(e) => {
        // A click on the backdrop lands on the <dialog> element itself.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog__inner">
        <header className="dialog__header">
          <p className="dialog__eyebrow">About this demo</p>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <IconClose size={18} />
          </button>
        </header>
        <div className="dialog__body prose">
          <h2 id="about-title">
            AMA-KBQA: An Adaptable Multi-Agent Framework for Generalizable Knowledge Base Question Answering
          </h2>
          <Markdown>{ABOUT_MD}</Markdown>
          <p className="muted">
            Click the <strong>?</strong> button in the corner anytime to reopen this panel.
          </p>
        </div>
      </div>
    </dialog>
  );
}
