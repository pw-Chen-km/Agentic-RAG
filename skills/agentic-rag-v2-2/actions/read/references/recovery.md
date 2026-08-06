# READ recovery

The previous READ parameters were rejected. Preserve the inherited intent and
correct the Chunk selection.

1. Inspect the typed Validator error.
2. Select one exact `chunk_id` from `legal_action_options` whose title or preview
   best matches the intent.
3. Do not repeat the rejected parameter object.

Return only the corrected READ parameter object. Do not switch action types.
