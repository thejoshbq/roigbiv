"""UI services — data loading, pipeline runner, color palettes, HITL state.

Keep Dash callbacks thin: they read/write process-local singletons defined
here rather than holding state directly.
"""
