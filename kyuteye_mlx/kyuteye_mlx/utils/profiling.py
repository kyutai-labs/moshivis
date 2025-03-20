from typing import Callable

import line_profiler

PROFILING_ENABLED = False
profile: line_profiler.LineProfiler | Callable
if PROFILING_ENABLED:
    profile = line_profiler.LineProfiler()
else:

    def profile(x: Callable) -> Callable:
        return x
