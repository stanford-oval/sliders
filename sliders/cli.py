"""Command-line entry point for SLIDERS.

Installed as the ``sliders`` console script via ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sliders",
        description="Run SLIDERS on one or more Markdown or PDF documents and answer a question about them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  sliders --docs paper.pdf --question "What are the key findings?"
  sliders --docs ./papers/ --question "Compare the results" --verbose
  sliders --docs a.md b.pdf --question "Summarize" --debug --output-dir ./out/

Programmatic usage:
  from sliders.run import run_sliders
  answer = run_sliders(docs="./papers/", question="What are the key findings?")
""",
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Path(s) to .md or .pdf file(s), or a single directory containing them. "
        "PDFs are auto-converted to markdown via Docling on the fly.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question to answer from the documents.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream pipeline logs to the terminal (default: only the final answer is printed).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Persist intermediate reconciliation tables as CSVs under the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output artifacts (default: ./sliders_output/<timestamp>/).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a custom SLIDERS YAML config (default: bundled default_config.yaml).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key. Passing this automatically routes to the OpenAI provider.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL (defaults to https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--azure-api-key",
        default=None,
        help="Azure OpenAI API key (falls back to AZURE_OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint (falls back to AZURE_OPENAI_ENDPOINT env var).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from sliders.run import run_sliders

    try:
        result = run_sliders(
            docs=args.docs,
            question=args.question,
            verbose=args.verbose,
            debug=args.debug,
            output_dir=args.output_dir,
            config_path=args.config,
            return_full_result=True,
            azure_api_key=args.azure_api_key,
            azure_endpoint=args.azure_endpoint,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.verbose:
        print()
    print(result["answer"])

    if args.verbose:
        print(f"\nFull results saved to: {result['results_json_path']}")
        if args.debug:
            print(f"Intermediate tables saved in: {result['output_dir']}")


if __name__ == "__main__":
    main()
