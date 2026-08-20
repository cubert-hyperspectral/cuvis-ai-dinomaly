"""Shared scaffolding for streaming metric nodes (val/test only).

Streaming metric nodes accumulate state across batches and must start fresh whenever a new
``(stage, epoch)`` begins. This base owns exactly that bookkeeping: it defaults execution to
VAL/TEST, tracks the last-seen ``(stage, epoch)`` key, and clears subclass state (via the
:meth:`_StreamingMetric._reset_state` hook) on the boundary or on an explicit :meth:`reset`.
Subclasses implement ``_reset_state`` and call :meth:`_reset_on_epoch_boundary` at the top of
``forward``.
"""

from __future__ import annotations

from typing import Any

from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context


class _StreamingMetric(Node):
    """Base for streaming metric nodes (defaults to VAL/TEST execution).

    Accumulates across batches within one ``(stage, epoch)``; subclasses store their
    accumulators, implement :meth:`_reset_state`, and open ``forward`` with
    :meth:`_reset_on_epoch_boundary` so each epoch starts from cleared state.
    """

    def __init__(self, **kwargs: Any) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(name=name, execution_stages=execution_stages, **kwargs)
        self._last_key: tuple[ExecutionStage, int] | None = None

    def _reset_state(self) -> None:
        """Reset every streaming accumulator (plus any per-run bookkeeping). Subclass hook."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset accumulators (called by the Predictor before a run, and by tests)."""
        self._reset_state()
        self._last_key = None

    def _reset_on_epoch_boundary(self, context: Context) -> None:
        """Reset once per ``(stage, epoch)`` so each epoch accumulates fresh."""
        key = (context.stage, context.epoch)
        if self._last_key != key:
            self._reset_state()
            self._last_key = key
