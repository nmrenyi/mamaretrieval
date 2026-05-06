"""Configuration helpers for mamaretrieval scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return config


def expand_path(path: str | Path) -> Path:
    """Expand a user-provided path without requiring the file to exist."""
    return Path(path).expanduser()


def corpus_chunks_path(config: dict[str, Any]) -> Path:
    """Return the configured corpus chunk path."""
    return expand_path(config["corpus"]["chunks_path"])


def configured_sources(config: dict[str, Any]) -> list[str]:
    """Return configured benchmark sources in tier order."""
    sources: list[str] = []
    for tier in config["source_tiers"].values():
        sources.extend(tier["sources"])
    return sources


def source_tier_map(config: dict[str, Any]) -> dict[str, str]:
    """Return a source_id -> tier_name mapping."""
    mapping: dict[str, str] = {}
    for tier_name, tier in config["source_tiers"].items():
        for source in tier["sources"]:
            mapping[source] = tier_name
    return mapping

