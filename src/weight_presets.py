# src/weight_presets.py
"""
Central definition of weight presets for the MCDAEngine.

Edit this file to customize or add new presets.
"""

from __future__ import annotations

from typing import Dict


# Type alias (only for IDE / clarity)
WeightPresets = Dict[str, Dict[str, float]]

# IMPORTANT:
# - Inner dict keys must match your feature names
#   (e.g. "first_visit", "delivery", "coverage", "cost", "sla", "speed", etc.)
# - Weights do NOT need to be normalized here; MCDAEngine will normalize.
WEIGHT_PRESETS: WeightPresets = {

    # 1) Fully balanced across all 6 criteria
    "balanced": {
        "first_visit": 0.17,
        "cost":        0.17,
        "coverage":    0.16,
        "delivery":    0.17,
        "sla":         0.16,
        "speed":       0.17,
    },

    # 2) Prioritize full service (1era visita + entrega + SLA + speed)
    "service_quality": {
        "first_visit": 0.25,
        "cost":        0.13,
        "coverage":    0.10,
        "delivery":    0.22,
        "sla":         0.17,
        "speed":       0.13,
    },

    # 3) Cost is the main driver (speed is minor)
    "cost_focus": {
        "first_visit": 0.10,
        "cost":        0.46,
        "coverage":    0.09,
        "delivery":    0.12,
        "sla":         0.13,
        "speed":       0.10,
    },

    # 4) Expand coverage nationally (speed small, coverage huge)
    "coverage_focus": {
        "first_visit": 0.10,
        "cost":        0.10,
        "coverage":    0.48,
        "delivery":    0.11,
        "sla":         0.11,
        "speed":       0.10,
    },

    # 5) Reliability (delivery + SLA + speed) as top priority
    "reliability_first": {
        "first_visit": 0.12,
        "cost":        0.08,
        "coverage":    0.08,
        "delivery":    0.30,
        "sla":         0.30,
        "speed":       0.12,
    },

    # 6) Strict service: 1era visita & entrega >>> cost, speed moderate
    "service_strict": {
        "first_visit": 0.32,
        "cost":        0.04,
        "coverage":    0.08,
        "delivery":    0.28,
        "sla":         0.18,
        "speed":       0.10,
    },

    # 7) SLA-focused (predictability/time commitments), speed second
    "sla_priority": {
        "first_visit": 0.12,
        "cost":        0.08,
        "coverage":    0.08,
        "delivery":    0.17,
        "sla":         0.40,
        "speed":       0.15,
    },

    # 8) Premium shipping (fast + reliable, cost less relevant)
    "premium_shipping": {
        "first_visit": 0.20,
        "cost":        0.07,
        "coverage":    0.06,
        "delivery":    0.25,
        "sla":         0.22,
        "speed":       0.20,
    },

    # 9) Startup scaling mode: reach + price (speed low but nonzero)
    "startup_scaling": {
        "first_visit": 0.08,
        "cost":        0.32,
        "coverage":    0.32,
        "delivery":    0.09,
        "sla":         0.09,
        "speed":       0.10,
    },

    # 10) Practical mix (good real-world baseline)
    "practical": {
        "first_visit": 0.21,
        "cost":        0.31,
        "coverage":    0.14,
        "delivery":    0.15,
        "sla":         0.09,
        "speed":       0.10,
    },
}


def get_weight_presets() -> WeightPresets:
    """
    Public accessor. You can swap implementation later
    (e.g. load from JSON/YAML) without touching the rest of the code.
    """
    # Return a shallow copy to avoid accidental in-place mutation
    return {name: preset.copy() for name, preset in WEIGHT_PRESETS.items()}
