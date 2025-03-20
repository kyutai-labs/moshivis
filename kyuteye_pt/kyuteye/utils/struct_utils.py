"""Useful structure and simple class definition

FrozenEnum are used to hold global configs shared across multiple files"""

from enum import Enum, EnumMeta
from typing import Any


class FrozenEnumMeta(EnumMeta):
    "Enum metaclass that freezes an enum entirely"

    def __new__(mcs, name: str, bases: Any, classdict: Any) -> type:
        classdict["__frozenenummeta_creating_class__"] = True
        enum = super().__new__(mcs, name, bases, classdict)
        del enum.__frozenenummeta_creating_class__  # type: ignore[attr-defined]
        return enum

    def __setattr__(cls, name: str, value: Any) -> None:
        members = cls.__dict__.get("_member_map_", {})
        if hasattr(cls, "__frozenenummeta_creating_class__") or name in members:
            return super().__setattr__(name, value)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))

    def __delattr__(cls, name: str) -> None:
        members = cls.__dict__.get("_member_map_", {})
        if hasattr(cls, "__frozenenummeta_creating_class__") or name in members:
            return super().__delattr__(name)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))


class FrozenEnum(Enum, metaclass=FrozenEnumMeta):
    """Frozen Enum type used for immutable configurations"""

    pass
