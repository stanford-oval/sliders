"""Programmatic entry point for running SLIDERS on arbitrary Markdown documents.

Example:
    >>> from sliders.run import run_sliders
    >>> answer = run_sliders(docs="./my_papers/", question="What are the key findings?")
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml


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


_SUPPORTED_SUFFIXES = {".md", ".pdf"}
_docling_converter = None


def _get_docling_converter():
    """Lazily create and cache a Docling DocumentConverter for the current process."""
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError(
                "Docling is required to convert PDFs. Install it with `uv add docling` "
                "or pass pre-converted .md files to run_sliders."
            ) from e

        print(
            "Initializing Docling for PDF conversion. "
            "First run downloads layout models (~400 MB); subsequent runs reuse the cache.",
            flush=True,
        )
        _docling_converter = DocumentConverter()
    return _docling_converter


def _convert_pdf_to_markdown(pdf_path: Path, out_dir: Path) -> Path:
    """Convert a PDF to a markdown file inside ``out_dir`` using Docling.

    Returns the path to the generated ``.md`` file.
    """
    converter = _get_docling_converter()
    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()
    out_path = out_dir / (pdf_path.stem + ".md")
    out_path.write_text(md_text)
    return out_path


def _resolve_docs_dir(docs: Union[str, list[str]], tmp_dir: str) -> str:
    """Normalize the ``docs`` argument into a directory of ``.md`` files.

    Accepts ``.md`` files directly, or ``.pdf`` files which are converted to
    markdown on the fly via Docling. Mixed inputs (some markdown, some PDF)
    are fine. If ``docs`` is a single directory containing only ``.md``
    files, it is used as-is; otherwise everything is staged into
    ``tmp_dir/docs`` (symlinked for ``.md``, converted for ``.pdf``).
    """
    if isinstance(docs, str):
        docs = [docs]

    if len(docs) == 1 and Path(docs[0]).is_dir():
        doc_dir = Path(docs[0]).resolve()
        md_files = list(doc_dir.glob("*.md"))
        pdf_files = list(doc_dir.glob("*.pdf"))
        if not md_files and not pdf_files:
            raise FileNotFoundError(f"No .md or .pdf files found in directory: {doc_dir}")
        if not pdf_files:
            return str(doc_dir)
        # Mixed/PDF directory: stage into tmp_dir
        docs = [str(p) for p in sorted(md_files + pdf_files)]

    for doc_path in docs:
        p = Path(doc_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {doc_path}")
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Expected .md or .pdf file, got: {doc_path}")

    stage_dir = Path(tmp_dir) / "docs"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for doc_path in docs:
        src = Path(doc_path).resolve()
        if src.suffix.lower() == ".pdf":
            print(f"Converting {src.name} to markdown...")
            _convert_pdf_to_markdown(src, stage_dir)
        else:
            dst = stage_dir / src.name
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)

    return str(stage_dir)


def _load_config(config_path: str | None) -> dict:
    """Load a SLIDERS YAML config, falling back to the bundled default."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")

    config_path = str(Path(config_path).resolve())
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def _run_sliders_async(
    docs: Union[str, list[str]],
    question: str,
    verbose: bool = False,
    debug: bool = False,
    output_dir: str | None = None,
    config_path: str | None = None,
    return_full_result: bool = False,
    azure_api_key: str | None = None,
    azure_endpoint: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    schema: Any = None,
) -> Union[str, dict]:
    """Async implementation of :func:`run_sliders`."""
    if azure_api_key or azure_endpoint or openai_api_key or openai_base_url:
        from sliders.llm.llm import set_llm_credentials

        set_llm_credentials(
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )

    if output_dir:
        out_dir = Path(output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.cwd() / "sliders_output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SLIDERS_RESULTS", str(out_dir))

    if not verbose:
        from sliders.log_utils import suppress_console_logging

        suppress_console_logging()

    from sliders.globals import SlidersGlobal
    from sliders.experiments.wiki_celeb import WikiCeleb
    from sliders.system import SlidersAgent

    config = _load_config(config_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        files_dir = _resolve_docs_dir(docs, tmp_dir)

        questions_file = os.path.join(tmp_dir, "questions.txt")
        with open(questions_file, "w") as f:
            f.write(question + "\n")

        config["experiment_config"]["questions_path"] = questions_file
        config["experiment_config"]["files_dir"] = files_dir
        config["experiment_config"]["num_questions"] = None

        if debug:
            config["system_config"].setdefault("merge_tables", {}).setdefault("reconciliation", {})["debug_mode"] = True

        if schema is not None:
            config["system_config"].setdefault("generate_schema", {})["user_schema"] = schema

        config["system_config"]["output_folder"] = str(out_dir)

        experiment = WikiCeleb(config["experiment_config"])
        system = SlidersAgent(config["system_config"])

        results = await experiment.run(
            system,
            sample_size=config["experiment_config"].get("num_questions"),
            random_state=config["experiment_config"].get("random_state", 42),
        )

        try:
            results["config"] = config
        except Exception:
            pass

        output_file = out_dir / config.get("output_file", "sliders_output.json").replace(
            ".json", f"_{SlidersGlobal.experiment_id}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        answers: list[str] = []
        for result in results.get("results", []):
            if "predicted_answer" in result:
                answers.append(result["predicted_answer"])
            elif "error" in result:
                answers.append(f"[Error] {result['error']}")

        answer = "\n\n".join(answers)

        if return_full_result:
            return {
                "answer": answer,
                "answers": answers,
                "results": results,
                "results_json_path": str(output_file),
                "output_dir": str(out_dir),
            }

        return answer


def run_sliders(
    docs: Union[str, list[str]],
    question: str,
    verbose: bool = False,
    debug: bool = False,
    output_dir: str | None = None,
    config_path: str | None = None,
    return_full_result: bool = False,
    azure_api_key: str | None = None,
    azure_endpoint: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    schema: Any = None,
) -> Union[str, dict]:
    """Run SLIDERS on a document corpus and return the answer to ``question``.

    Args:
        docs: A directory of ``.md`` / ``.pdf`` files, a single file path, or a
            list of file paths. PDFs are auto-converted to Markdown via Docling.
        question: Natural-language question to answer against the corpus.
        verbose: If ``True``, stream pipeline logs to the terminal.
        debug: If ``True``, persist intermediate reconciliation tables as CSVs.
        output_dir: Directory for output artifacts. Defaults to
            ``./sliders_output/<timestamp>/``.
        config_path: Path to a custom YAML config. Defaults to
            ``configs/default_sliders.yaml``.
        return_full_result: If ``True``, return a dict with the answer, raw
            results, and artifact paths instead of just the answer string.
        azure_api_key: Azure OpenAI API key; falls back to
            ``AZURE_OPENAI_API_KEY``.
        azure_endpoint: Azure OpenAI endpoint; falls back to
            ``AZURE_OPENAI_ENDPOINT``.
        openai_api_key: OpenAI API key. Passing this automatically switches the
            current call to the public OpenAI provider; falls back to
            ``OPENAI_API_KEY`` when ``SLIDERS_LLM_PROVIDER=openai``.
        openai_base_url: OpenAI-compatible base URL (e.g. for Azure AI Foundry
            or a self-hosted gateway); defaults to ``https://api.openai.com/v1``.
        schema: Optional user-provided schema. Accepts a list of table dicts or
            a ``{"tables": [...]}`` dict. Each table has a ``name``, an
            optional ``description``, and a ``fields`` list where every field
            is a string (field name) or a dict with ``name`` and any of
            ``data_type``, ``description``, ``required``, ``unit``, ``scale``.
            Any missing metadata is filled in by a single LLM call; the LLM
            will not add tables or fields the user did not list. Passing a
            schema bypasses schema induction entirely.

    Returns:
        The predicted answer as a string, or (when ``return_full_result=True``)
        a dict with the keys ``answer``, ``answers``, ``results``,
        ``results_json_path``, and ``output_dir``.
    """
    return asyncio.run(
        _run_sliders_async(
            docs=docs,
            question=question,
            verbose=verbose,
            debug=debug,
            output_dir=output_dir,
            config_path=config_path,
            return_full_result=return_full_result,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            schema=schema,
        )
    )
