"""Pluggable speech enhancement model backends.

All models implement :class:`ai.models.base.EnhancementModel` so the
streaming engine and hardware integration code depend only on the interface,
never on a specific model implementation.
"""

from ai.models.base import EnhancementModel

__all__ = ["EnhancementModel"]
