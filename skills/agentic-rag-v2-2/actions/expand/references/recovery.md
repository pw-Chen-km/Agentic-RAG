# EXPAND recovery

The previous EXPAND parameters were rejected. Preserve the inherited intent and
correct only the invalid fields.

1. Inspect the typed Validator error.
2. Select one exact combination of `kind`, `source_id`, and `direction` from
   `legal_action_options`.
3. Use the visible source's semantic label or text to keep any query aligned
   with the inherited intent.
4. Use `top_k=5` and do not repeat the rejected parameter object.

Return only the corrected EXPAND parameter object. Do not switch action types.
