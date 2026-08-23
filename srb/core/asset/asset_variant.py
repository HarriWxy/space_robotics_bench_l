from enum import Enum, auto

from typing_extensions import Self


class AssetVariant(str, Enum):
    PRIMITIVE = "primitive"
    DATASET = "dataset"
    PROCEDURAL = "procedural"

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str) -> Self | None:
        return next(
            (variant for variant in cls if string.upper() == variant.name), None
        )
