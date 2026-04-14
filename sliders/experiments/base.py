"""Abstract base class for benchmark drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sliders.system import System


class Experiment(ABC):
    """Base class for benchmark drivers.

    Subclasses load their benchmark data in ``__init__`` and implement
    :meth:`_run_row` (per-question execution) and :meth:`run` (orchestration
    over the full question set).
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def _run_row(self, row: dict, system: "System", all_metadata: list, *args, **kwargs) -> dict:
        """Run a single question through ``system`` and return its result row."""

    @abstractmethod
    async def run(self, system: "System", parallel: bool = False, *args, **kwargs) -> dict:
        """Run the full benchmark through ``system`` and return aggregated results."""
