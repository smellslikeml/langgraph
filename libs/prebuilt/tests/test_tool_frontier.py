"""Tests for causal minimal tool filtering and its create_react_agent wiring."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool as dec_tool

from langgraph.prebuilt import CausalToolFilter, ToolContract
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_frontier import (
    infer_satisfied_facts,
    select_tool_frontier,
)
from tests.model import FakeToolCallingModel


@dec_tool
def search(query: str) -> str:
    """Search for documents."""
    return "results"


@dec_tool
def open_doc(doc_id: str) -> str:
    """Open a document."""
    return "page"


@dec_tool
def summarize(text: str) -> str:
    """Summarize text."""
    return "summary"


TOOLS = [search, open_doc, summarize]

CONTRACTS = {
    "search": ToolContract(provides=frozenset({"results"})),
    "open_doc": ToolContract(
        requires=frozenset({"results"}), provides=frozenset({"page"})
    ),
    "summarize": ToolContract(
        requires=frozenset({"page"}), provides=frozenset({"summary"})
    ),
}


def _names(tools: Any) -> set[str]:
    return {t.name for t in tools}


# --- Unit-level behavior of the frontier selection ---------------------------


def test_select_frontier_starts_minimal() -> None:
    # With nothing satisfied yet, only the unconditioned tool is exposed.
    frontier = select_tool_frontier(TOOLS, CONTRACTS, frozenset())
    assert _names(frontier) == {"search"}


def test_select_frontier_advances_with_facts() -> None:
    # After `results` is established, `search` is unnecessary and `open_doc`
    # becomes available, but `summarize` is still premature.
    frontier = select_tool_frontier(TOOLS, CONTRACTS, frozenset({"results"}))
    assert _names(frontier) == {"open_doc"}


def test_infer_facts_from_tool_messages() -> None:
    messages = [
        HumanMessage(content="hi"),
        ToolMessage(content="results", name="search", tool_call_id="1"),
    ]
    facts = infer_satisfied_facts(messages, CONTRACTS)
    assert facts == frozenset({"results"})


def test_infer_facts_ignores_errored_tools() -> None:
    messages = [
        ToolMessage(content="boom", name="search", tool_call_id="1", status="error"),
    ]
    assert infer_satisfied_facts(messages, CONTRACTS) == frozenset()


def test_uncontracted_tool_kept_by_default() -> None:
    @dec_tool
    def helper(x: str) -> str:
        """Helper without a contract."""
        return x

    frontier = select_tool_frontier([helper], {}, frozenset())
    assert _names(frontier) == {"helper"}
    # ...and can be dropped when keep_uncontracted is False.
    frontier = select_tool_frontier([helper], {}, frozenset(), keep_uncontracted=False)
    assert frontier == []


# --- Integration: the filter narrows tools per step inside create_react_agent -


class RecordingToolModel(FakeToolCallingModel):
    """A fake model that records which tools were bound on each step."""

    tools_seen: list[set[str]] = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        bound = kwargs.get("tools") or []
        self.tools_seen.append({t.get("name") or t["function"]["name"] for t in bound})
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_create_react_agent_exposes_minimal_frontier_per_step() -> None:
    model = RecordingToolModel(
        tools_seen=[],
        tool_calls=[
            [{"name": "search", "args": {"query": "q"}, "id": "1"}],
            [{"name": "open_doc", "args": {"doc_id": "d"}, "id": "2"}],
            [{"name": "summarize", "args": {"text": "t"}, "id": "3"}],
            [],
        ],
    )
    tool_filter = CausalToolFilter(CONTRACTS)

    agent = create_react_agent(model, TOOLS, tool_filter=tool_filter)
    agent.invoke({"messages": [HumanMessage(content="go")]})

    # The model only ever saw a single, advancing tool at each step.
    assert model.tools_seen[0] == {"search"}
    assert model.tools_seen[1] == {"open_doc"}
    assert model.tools_seen[2] == {"summarize"}


def test_create_react_agent_without_filter_exposes_all_tools() -> None:
    model = RecordingToolModel(
        tools_seen=[],
        tool_calls=[[]],
    )
    agent = create_react_agent(model, TOOLS)
    agent.invoke({"messages": [HumanMessage(content="go")]})

    assert model.tools_seen[0] == {"search", "open_doc", "summarize"}


def test_tool_filter_rejected_for_dynamic_model() -> None:
    def select_model(state: Any, runtime: Any) -> Any:
        return FakeToolCallingModel()

    try:
        create_react_agent(select_model, TOOLS, tool_filter=CausalToolFilter(CONTRACTS))
    except ValueError as exc:
        assert "static model" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("expected ValueError for dynamic model + tool_filter")
