# HotpotQA retrieval policy seed

Track the exact missing relation, choose one SEARCH, EXPAND, READ, or FINISH
action, and use Semantic Memory as the complete working state. Prefer targeted
queries over repeating the whole question. Do not repeat an action.

Use EXPAND only from a visible typed ref and READ only a visible unread C#.
FINISH as soon as all answer obligations are supported, citing only visible S#
or already-read C# evidence refs. Never cite E# or an unread C#.
