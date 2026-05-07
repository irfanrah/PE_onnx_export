from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

from ..config import PEInitConfig


class LoadStrategy(ABC):
    @abstractmethod
    def validate(self, cfg: PEInitConfig) -> None: ...

    @abstractmethod
    def build_model(self, cfg: PEInitConfig, load_clip: Callable[..., Any]) -> Any: ...
