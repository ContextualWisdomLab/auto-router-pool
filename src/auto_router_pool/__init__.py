"""Auto-router pool refresher.

Keeps the LiteLLM complexity routers pinned to the best index-per-dollar model
that is *currently* valid, and re-reads Artificial Analysis only when the
catalogue changes.
"""

__version__ = "0.1.0"
