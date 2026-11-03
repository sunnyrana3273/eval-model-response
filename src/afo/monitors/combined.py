"""Combined monitor: behavioural features concatenated with one layer's activations."""

from afo.monitors.base import LogisticMonitor


class CombinedMonitor(LogisticMonitor):
    name = "combined"
    uses_behavioral = True
    uses_activations = True
