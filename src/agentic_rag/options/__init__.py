"""Stateful, two-level Agentic-RAG Options runtime.

This package is deliberately separate from the legacy one-action controller.  An
episode first selects an option (a sub-goal) and then asks the selected option
for one primitive action.  After the option reports completion or blockage,
control returns to the selector.
"""

from agentic_rag.options.catalog import OptionCatalog, OptionSpec
from agentic_rag.options.controller import OptionsAgentController
from agentic_rag.options.harness import OptionsAgentHarness
from agentic_rag.options.models import (
    OptionEvent,
    OptionPolicyDecision,
    OptionSelectorDecision,
)

__all__ = [
    "OptionCatalog",
    "OptionSpec",
    "OptionEvent",
    "OptionPolicyDecision",
    "OptionSelectorDecision",
    "OptionsAgentController",
    "OptionsAgentHarness",
]
