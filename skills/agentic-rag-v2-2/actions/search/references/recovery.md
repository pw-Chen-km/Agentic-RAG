# SEARCH recovery

The previous SEARCH parameters were rejected. Preserve the inherited intent and
correct only the invalid fields.

1. Inspect the typed Validator error.
2. Select one exact method-target pair from `legal_action_options`.
3. Use a nonblank semantic query and `top_k=5`.
4. Do not repeat the rejected parameter object.

Return only the corrected SEARCH parameter object. Do not switch action types.
