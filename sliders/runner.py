"""Benchmark runner for SLIDERS and its baselines.

This script wires the ``experiment`` and ``system`` entries from a YAML config
into a concrete benchmark driver and baseline system, runs the experiment
end-to-end, and writes the results JSON under ``$SLIDERS_RESULTS``.

Example:
    uv run sliders/runner.py --config configs/benchmarks/finance_bench_sliders.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np
import yaml
from pydantic import BaseModel

from sliders.baselines import (
    ChainOfAgentsSystem,
    LLMSequentialSystem,
    LLMWithoutToolUseSystem,
    LLMWithToolUseSystem,
    QuestionGuidedBaselineSystem,
    RLMSystem,
)
from sliders.experiment import print_result_summary
from sliders.experiments.finance_bench import FinanceBench
from sliders.experiments.loong import Loong
from sliders.experiments.oolong import OoLong
from sliders.experiments.sec_10q import SEC10Q
from sliders.experiments.wiki_celeb import WikiCeleb
from sliders.globals import SlidersGlobal
from sliders.log_utils import logger
from sliders.system import SlidersAgent


SLIDERS_RESULTS = os.environ.get("SLIDERS_RESULTS", "./sliders_results")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes NumPy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SLIDERS benchmark config end-to-end.")
    parser.add_argument("--config", type=str, required=True, help="Path to a benchmark YAML config.")
    parser.add_argument("--parallel", action="store_true", help="Run questions in parallel.")
    return parser.parse_args()


class Config(BaseModel):
    """Validated shape of a benchmark YAML config."""

    system: str
    experiment: str
    config_file: str
    system_config: dict
    experiment_config: dict
    output_file: str

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            config["config_file"] = file_path
            return cls.model_validate(config)


EXPERIMENT_REGISTRY = {
    "finance_bench": FinanceBench,
    "loong": Loong,
    "oolong": OoLong,
    "sec_10q": SEC10Q,
    "wiki_celeb": WikiCeleb,
}

SYSTEM_REGISTRY = {
    "direct_tool_use": LLMWithToolUseSystem,
    "direct_no_tool_use": LLMWithoutToolUseSystem,
    "sequential": LLMSequentialSystem,
    "sliders": SlidersAgent,
    "rlm": RLMSystem,
    "question_guided": QuestionGuidedBaselineSystem,
    "chain_of_agents": ChainOfAgentsSystem,
}


async def run_experiment(config_file: str, parallel: bool = False) -> None:
    """Load a benchmark config, run it, and write the results JSON."""
    config = Config.from_file(config_file)
    experiment = EXPERIMENT_REGISTRY[config.experiment](config.experiment_config)
    system = SYSTEM_REGISTRY[config.system](config.system_config)

    results = await experiment.run(
        system,
        sample_size=config.experiment_config.get("num_questions"),
        random_state=config.experiment_config.get("random_state", 42),
        parallel=parallel,
        qa_type=config.experiment_config.get("qa_type"),
    )

    try:
        results["config"] = config.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Error dumping config: {e}")

    print_result_summary(results)

    output_file = os.path.join(
        SLIDERS_RESULTS,
        config.output_file.replace(".json", f"_{SlidersGlobal.experiment_id}.json"),
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_experiment(args.config, args.parallel))
    except Exception as e:
        logger.error(e)
        import traceback

        traceback.print_exc()
