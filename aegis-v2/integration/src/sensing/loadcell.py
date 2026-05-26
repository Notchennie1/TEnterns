"""
Load Cell Reader
================
Reads bin weights and the physical shelf layout (number of layers/rows and
bins per layer) from load-cell hardware.

⚠️  NOT YET IMPLEMENTED — hardware integration is pending.

This is a stub that satisfies the interface the rest of the system expects so
the pipeline and dashboard run unchanged today. Every reader returns empty /
zero values, which the state layer merges with the CV-detected layout (see
``PipelineState.get_layout``). When the real hardware lands, fill in
``get_layout`` / ``get_weights`` and flip ``is_connected`` — no callers change.

Layout model
------------
A "layer" is one horizontal row of bins (the ``row`` in ``bin_{row}_{col}``).
``LayerLayout.bins_per_layer`` maps each layer index to how many bins that
layer contains, as reported by the load cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("aegis.sensing.loadcell")


@dataclass
class LayerLayout:
    """Physical bin layout as reported by the load cells."""
    num_layers: int = 0
    bins_per_layer: dict[int, int] = field(default_factory=dict)  # layer index -> bin count


class LoadCellReader:
    """
    Reads weights and layer layout from the load-cell array.

    Stub implementation — returns no layers and no weights. The merge logic in
    :class:`~integration.src.ui.state.PipelineState` falls back to the
    CV-detected layout whenever the load cells report nothing, so the dashboard
    still renders bins from vision alone.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._enabled = bool(self._config.get("enabled", False))
        if self._enabled:
            # TODO: open the serial / Modbus connection to the load-cell array.
            logger.warning("Load cells marked enabled in config but driver is not implemented yet")

    def is_connected(self) -> bool:
        """True once real hardware is talking. Always False in the stub."""
        return False

    def get_layout(self) -> LayerLayout:
        """
        Return the bin layout sensed by the load cells.

        Stub: returns an empty layout, so :meth:`PipelineState.get_layout`
        uses the CV-detected grid instead.
        """
        return LayerLayout()

    def get_weights(self) -> dict[str, float]:
        """
        Return current weight per bin in grams, keyed by ``bin_{row}_{col}``.

        Stub: returns an empty mapping.
        """
        return {}

    def close(self) -> None:
        """Release the hardware connection (no-op in the stub)."""
        pass
