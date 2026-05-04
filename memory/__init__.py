# memory/__init__.py
from .long_term import get_long_term_memory, close_long_term_memory
from .short_term import get_short_term_memory, close_short_term_memory

__all__ = [
    "get_long_term_memory",
    "close_long_term_memory",
    "get_short_term_memory",
    "close_short_term_memory",
]