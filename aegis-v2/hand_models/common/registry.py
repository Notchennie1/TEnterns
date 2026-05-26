"""
Hand Tracker Registry
=====================
Service-locator pattern: the integration pipeline instantiates any
registered hand-tracking backend by name from config.

Add new backends by calling:
    TrackerRegistry.register("my_backend", MyTrackerClass)
"""

from __future__ import annotations

import logging
from typing import Type

from .base_hand_tracker import BaseHandTracker

logger = logging.getLogger("hand_models.registry")


class TrackerRegistry:
    """Central registry of hand-tracker backends."""

    _backends: dict[str, Type[BaseHandTracker]] = {}

    @classmethod
    def register(cls, name: str, tracker_class: Type[BaseHandTracker]) -> None:
        cls._backends[name.lower()] = tracker_class
        logger.debug("Registered hand tracker: %s", name)

    @classmethod
    def create(cls, name: str, config: dict) -> BaseHandTracker:
        """Instantiate a registered backend by name and load its model."""
        key = name.lower()
        if key not in cls._backends:
            available = ", ".join(cls._backends.keys()) or "(none)"
            raise ValueError(
                f"Unknown hand tracker '{name}'. Available: {available}"
            )
        tracker = cls._backends[key](config)
        tracker.load_model()
        return tracker

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._backends.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[BaseHandTracker] | None:
        return cls._backends.get(name.lower())
