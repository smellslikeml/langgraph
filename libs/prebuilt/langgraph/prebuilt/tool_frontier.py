"""Causal minimal tool filtering for tool-calling agents.

This module computes the *minimal next-step tool frontier* for an agent: the
smallest set of tools whose preconditions are satisfied by the current state
and that still advance the task toward the goal. Exposing only the frontier to
the model (instead of the full tool menu) reduces wrong-tool calls, premature
actions, and token cost, while keeping execution traces focused on the tool
that was actually causally relevant at each step.

Tools are described by lightweight precondition-effect contracts:

- `requires`: facts that must hold in the current state before the tool is
  useful (an unmet precondition means the tool is *premature*).
- `provides`: facts the tool establishes once it has run successfully (if all
  of a tool's effects already hold, the tool is *unnecessary* and is dropped).

Satisfied facts are inferred from the agent's message history: a tool's effects
become satisfied once a successful `ToolMessage` for that tool appears. This is
training-free and requires no extra model calls.

Adapted from "ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable
LLM Agents" (arXiv:2606.06284). We implement the causal-sufficiency selection
contract (tools-in / filtered-tools-out) rather than the paper's full
benchmark harness; the contract is what slots into an agent's per-step tool
binding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

__all__ = [
    "ToolContract",
    "ToolFilter",
    "CausalToolFilter",
    "select_tool_frontier",
    "infer_satisfied_facts",
]

# A tool filter takes the agent's full tool list plus the current graph state
# and returns the subset to expose to the model for the next step.
ToolFilter = Callable[[Sequence[BaseTool], Any], Sequence[BaseTool]]


@dataclass(frozen=True)
class ToolContract:
    """A lightweight precondition-effect contract for a single tool.

    Args:
        requires: Facts that must already hold for the tool to be applicable.
            If any required fact is missing, the tool is considered premature
            and is excluded from the frontier.
        provides: Facts established once the tool has run successfully. When all
            of a tool's effects are already satisfied, the tool is considered
            unnecessary and is excluded from the frontier.
    """

    requires: frozenset[str] = field(default_factory=frozenset)
    provides: frozenset[str] = field(default_factory=frozenset)


# Accepted shorthand for a contract: a ToolContract, a mapping with
# `requires`/`provides` keys, or a `(requires, provides)` tuple.
ContractLike = (
    ToolContract | Mapping[str, Iterable[str]] | tuple[Iterable[str], Iterable[str]]
)


def _coerce_contract(value: ContractLike) -> ToolContract:
    if isinstance(value, ToolContract):
        return value
    if isinstance(value, Mapping):
        return ToolContract(
            requires=frozenset(value.get("requires", ())),
            provides=frozenset(value.get("provides", ())),
        )
    requires, provides = value
    return ToolContract(requires=frozenset(requires), provides=frozenset(provides))


def _state_messages(state: Any) -> Sequence[Any]:
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    return messages or []


def infer_satisfied_facts(
    messages: Sequence[Any],
    contracts: Mapping[str, ToolContract],
    initial_facts: Iterable[str] = (),
) -> frozenset[str]:
    """Infer which facts hold given the tool calls already in the history.

    Starting from `initial_facts`, every successful `ToolMessage` contributes
    the `provides` facts of its contract. Tool messages with an error status do
    not contribute effects, since the action did not succeed.
    """
    satisfied: set[str] = set(initial_facts)
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "status", "success") == "error":
            continue
        contract = contracts.get(message.name)
        if contract is not None:
            satisfied |= contract.provides
    return frozenset(satisfied)


def select_tool_frontier(
    tools: Sequence[BaseTool],
    contracts: Mapping[str, ToolContract],
    satisfied: frozenset[str],
    keep_uncontracted: bool = True,
) -> list[BaseTool]:
    """Return the minimal next-step frontier from `tools`.

    A tool is in the frontier when its preconditions are met and at least one
    of its effects is not yet satisfied. Tools without a contract are kept when
    `keep_uncontracted` is true (the conservative default), since we cannot
    reason about their causal role.
    """
    frontier: list[BaseTool] = []
    for tool in tools:
        contract = contracts.get(tool.name)
        if contract is None:
            if keep_uncontracted:
                frontier.append(tool)
            continue
        # Premature: a required fact is not yet established.
        if not contract.requires <= satisfied:
            continue
        # Unnecessary: every effect this tool would produce already holds.
        if contract.provides and contract.provides <= satisfied:
            continue
        frontier.append(tool)
    return frontier


class CausalToolFilter:
    """A reusable `ToolFilter` built from precondition-effect contracts.

    Example:
        ```python
        tool_filter = CausalToolFilter(
            {
                "search": ToolContract(provides={"results"}),
                "open": ToolContract(requires={"results"}, provides={"page"}),
                "summarize": ToolContract(requires={"page"}, provides={"summary"}),
            }
        )
        agent = create_react_agent(model, tools, tool_filter=tool_filter)
        ```

    Before any tool has run, only `search` is exposed; after a successful
    `search`, `open` becomes available, and so on. This is the
    tools-in / filtered-tools-out contract from the CMTF paper.

    Args:
        contracts: Mapping from tool name to a `ToolContract` (or a mapping /
            `(requires, provides)` tuple coerced into one).
        initial_facts: Facts assumed to hold before the agent starts.
        keep_uncontracted: Whether to keep tools that have no contract.
    """

    def __init__(
        self,
        contracts: Mapping[str, ContractLike],
        *,
        initial_facts: Iterable[str] = (),
        keep_uncontracted: bool = True,
    ) -> None:
        self.contracts: dict[str, ToolContract] = {
            name: _coerce_contract(value) for name, value in contracts.items()
        }
        self.initial_facts = frozenset(initial_facts)
        self.keep_uncontracted = keep_uncontracted

    def __call__(self, tools: Sequence[BaseTool], state: Any) -> list[BaseTool]:
        satisfied = infer_satisfied_facts(
            _state_messages(state), self.contracts, self.initial_facts
        )
        return select_tool_frontier(
            tools, self.contracts, satisfied, self.keep_uncontracted
        )
