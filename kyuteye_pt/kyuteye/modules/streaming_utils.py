# pylint: disable=protected-access
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Common API for streaming modules during inference"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

import torch

State = Dict[str, float | int | torch.Tensor]


class StreamingModule(torch.nn.Module):
    """Common API for streaming components."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = {}
        self._is_streaming = False

    @property
    def empty_streaming_state(self) -> bool:
        """whether streaming state is empty"""
        return len(self._streaming_state) == 0

    def has_streaming_attribute(self, key: str) -> bool:
        """Whether `key` exists in the current streaming state"""
        return self._is_streaming and key in self._streaming_state

    def add_streaming_attribute(
        self, key: str, value: float | int | torch.Tensor
    ) -> None:
        """Add `value` into streaming state's `key`"""
        self._streaming_state[key] = value

    def get_streaming_attribute(self, key: str, default: Any = None) -> Any:
        """Add `value` into streaming state's `key`"""
        return self._streaming_state.get(key, default)

    @property
    def is_streaming(self) -> bool:
        """in streaming mode"""
        return self._is_streaming

    def get_streaming_info_as_int(self, attr_name: str, default: int = 0) -> int:
        """Tries to get attr_name as an integer"""
        if self._is_streaming and attr_name in self._streaming_state:
            if isinstance(self._streaming_state[attr_name], int):
                return self._streaming_state[attr_name]  # type: ignore
            if isinstance(self._streaming_state[attr_name], torch.Tensor):
                return int(self._streaming_state[attr_name].item())  # type: ignore
            raise ValueError(
                f"Unexpected type {type(self._streaming_state[attr_name])} in streaming state"
            )
        return default

    @property
    def streaming_offset(self) -> int:
        """Shortcut to get the current temporal offset in streaming mode"""
        return self.get_streaming_info_as_int("offset", default=0)

    @streaming_offset.setter
    def streaming_offset(self, value: int | torch.Tensor) -> None:
        if not self._is_streaming:
            raise NotImplementedError(
                "Updating streaming offset of a non-streaming module"
            )
        self._streaming_state["offset"] = value  # type: ignore

    def _apply_named_streaming(self, fn: Callable) -> None:
        for name, module in self.named_modules():
            if isinstance(module, StreamingModule):
                fn(name, module)

    def _set_streaming(self, streaming: bool) -> None:
        def _set_streaming(_: str, module: StreamingModule) -> None:
            module._is_streaming = streaming

        self._apply_named_streaming(_set_streaming)

    @contextmanager
    def streaming(self) -> Iterator:
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._set_streaming(True)
        try:
            yield
        finally:
            self._set_streaming(False)
            self.reset_streaming()

    def streaming_forever(self, batch_size: Optional[int] = None) -> None:
        """Set in permanent streaming state"""
        del batch_size
        self._set_streaming(True)

    def reset_streaming(self) -> None:
        """Reset the streaming state."""

        def _reset(_: str, module: StreamingModule) -> None:
            module._streaming_state.clear()

        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> State:
        """Return the streaming state, including that of sub-modules."""
        state: State = {}

        def _add(name: str, module: StreamingModule) -> None:
            if name:
                name += "."
            for key, value in module._streaming_state.items():
                state[name + key] = value

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: State) -> None:
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule) -> None:
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                # complexity is not ideal here, but probably fine.
                if key.startswith(name):
                    local_key = key[len(name) :]
                    if "." not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        assert len(state) == 0, list(state.keys())

    def flush(self, x: Optional[torch.Tensor] = None) -> Optional["StreamingModule"]:
        """Flush any remaining outputs that were waiting for completion.
        Typically, for convolutions, this will add the final padding
        and process the last buffer.

        This should take an optional argument `x`, which will be provided
        if a module before this one in the streaming pipeline has already
        spitted out a flushed out buffer.
        """
        if x is None:
            return None
        return self(x)
