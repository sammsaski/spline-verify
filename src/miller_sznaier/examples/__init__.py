"""Examples and comparison scripts for Miller-Sznaier distance estimation."""

from .flow_system import run_flow_example
from .comparison import compare_methods, ComparisonResult

__all__ = [
    'run_flow_example',
    'compare_methods',
    'ComparisonResult',
]
