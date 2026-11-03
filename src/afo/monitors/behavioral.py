"""Behavioural baseline: logistic regression on observable trajectory features."""

from afo.monitors.base import LogisticMonitor


class BehavioralMonitor(LogisticMonitor):
    name = "behavioral"
    uses_behavioral = True
