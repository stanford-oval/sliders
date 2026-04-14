import os
import re
from decimal import Decimal, InvalidOperation
from typing import Callable
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from sliders.document import contextualize_document_metadata
from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.globals import SlidersGlobal
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.llm_models import NumericExtraction
from sliders.log_utils import logger
from sliders.experiments.base import Experiment


SOFT_EVALUATOR_TOOL_NAME = "LLMAsJudgeEvaluationTool_soft_evaluator"


def log_oolong_results(result):
    if result.get("error"):
        logger.warning(f"Evaluation skipped due to error: {result['error']}")
        return
    if "gold_answer" not in result or "predicted_answer" not in result:
        logger.warning("Result missing answer fields; skipping detailed logging.")
        return
    logger.info(f"Gold Answer: {result['gold_answer']}")
    logger.info(f"Predicted Answer: {result['predicted_answer']}")
    for key, value in result.get("evaluation_tools", {}).items():
        # Handle different evaluation result formats
        if isinstance(value, dict):
            if "correct" in value:
                logger.info(f"{key}: {value['correct']}")
            else:
                logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {value}")


class OoLong(Experiment):
    """Driver for the Oolong (Bertsch et al., 2025) long-context aggregation benchmark.

    Uses the Oolong-Synth subset. Each instance pairs a single long document
    with a question that requires local classification followed by a global
    aggregation. Non-numeric answers are scored with an LLM judge; numeric
    answers use Oolong's deviation-aware metric.
    """

    def __init__(self, config: dict):
        self.config = config

        benchmark_path = self._resolve_benchmark_path()
        self.dataset = Dataset(benchmark_path)
        self.context_text_field = self.config.get("context_text_field", "context_window_text")
        self._question_ids = self._parse_question_ids()
        self.numeric_eval_enabled = self.config.get("enable_numeric_scoring", True)
        self.numeric_eval_prompt_file = self.config.get(
            "numeric_extractor_prompt", "evaluators/numeric_extractor.prompt"
        )
        numeric_model = (
            self.config.get("numeric_extractor_model")
            or self.config.get("hard_evaluator_model")
            or self.config.get("soft_evaluator_model")
        )
        self.numeric_eval_model = numeric_model
        self.numeric_eval_temperature = self.config.get("numeric_extractor_temperature", 0.0)
        self.numeric_extractor_chain = self._build_numeric_extractor_chain()

        self.benchmark_path = benchmark_path
        self.evaluator = Evaluator()

        # Evaluation tools
        # Soft evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                model=self.config["soft_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        # Hard evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

    def _resolve_benchmark_path(self) -> str:
        benchmark_path = self.config.get("benchmark_path")
        if benchmark_path:
            return benchmark_path

        benchmark_dir = self.config.get("benchmark_dir", "/path/to/datasets/oolong")
        dataset_variant = self.config.get("dataset_variant", "synth")
        split = self.config.get("split", "validation")
        context_len = self.config.get("context_len")

        base_name = f"oolongbench_oolong-{dataset_variant}_{split}"
        if dataset_variant == "synth":
            if context_len is None:
                raise ValueError(
                    "context_len must be specified in the config when using the synthetic OoLong benchmark"
                )
            base_name = f"{base_name}_contextlen_{context_len}"
        filename = f"{base_name}.json"

        benchmark_path = os.path.join(benchmark_dir, filename)
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Resolved benchmark path does not exist: {benchmark_path}")
        return benchmark_path

    def _parse_question_ids(self) -> list[str] | None:
        # Priority: specific_ids_csv takes precedence
        if self.config.get("specific_ids_csv"):
            try:
                import pandas as pd

                id_sample_df = pd.read_csv(self.config["specific_ids_csv"], comment="#")
                specific_ids = [str(qid) for qid in id_sample_df["id"].tolist()]
                logger.info(f"Loaded {len(specific_ids)} question IDs from CSV: {self.config['specific_ids_csv']}")
                return specific_ids
            except Exception as e:
                logger.warning(f"Failed to load specific IDs CSV: {e}, falling back to question_ids config")

        question_ids = self.config.get("question_ids")
        if question_ids is None:
            single_question_id = self.config.get("question_id")
            if single_question_id is None:
                return None
            question_ids = [single_question_id]

        if isinstance(question_ids, (str, int)):
            normalized = [str(question_ids)]
        elif isinstance(question_ids, (list, tuple, set)):
            normalized = [str(qid) for qid in question_ids if qid is not None]
        else:
            raise ValueError("question_ids must be provided as a string, integer, or iterable of IDs.")

        return normalized or None

    def _build_numeric_extractor_chain(self):
        if not self.numeric_eval_enabled:
            return None
        if not self.numeric_eval_model:
            logger.warning(
                "Numeric evaluation enabled but no numeric extractor model configured; disabling numeric scoring."
            )
            self.numeric_eval_enabled = False
            return None

        try:
            llm_client = get_llm_client(model=self.numeric_eval_model, temperature=self.numeric_eval_temperature)
            prompt_template = load_fewshot_prompt_template(self.numeric_eval_prompt_file, [])
            return prompt_template | llm_client.with_structured_output(NumericExtraction)
        except Exception as exc:
            logger.warning(f"Failed to initialize numeric extractor chain: {exc}. Numeric scoring disabled.")
            self.numeric_eval_enabled = False
            return None

    @property
    def description(self) -> str:
        return self.config.get(
            "document_description",
            "Differnt text chunks for answering the question. The labels are not provided, you have to infer the labels from the context. Each row contains a different datapoint.",
        )

    def _should_use_numeric_eval(self, row: dict) -> bool:
        if not self.numeric_eval_enabled or self.numeric_extractor_chain is None:
            return False

        for key in ("evaluation_type", "answer_type"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                if "NUMERIC" in value.upper():
                    return True
            elif isinstance(value, (list, tuple, set)):
                normalized_values = [str(item).upper() for item in value if item is not None]
                if any("NUMERIC" in item for item in normalized_values):
                    return True
            else:
                if "NUMERIC" in str(value).upper():
                    return True
        return False

    async def _extract_numeric_answer(self, question: str, predicted_answer: str) -> str | None:
        if not predicted_answer:
            return None
        if self.numeric_extractor_chain is None:
            return None

        try:
            extraction = await self.numeric_extractor_chain.ainvoke(
                {"question": question, "predicted_answer": predicted_answer}
            )
        except Exception as exc:
            logger.warning(f"Numeric extraction failed: {exc}")
            return None

        extracted_value = getattr(extraction, "extracted_value", None)
        if extracted_value is None:
            return None
        extracted_str = str(extracted_value).strip()
        return extracted_str or None

    @staticmethod
    def _normalize_numeric_value(value: str | int | float) -> int:
        if value is None:
            raise ValueError("Cannot normalize a null numeric value.")
        if isinstance(value, (int, float)):
            return int(value)

        sanitized = str(value).strip()
        if not sanitized:
            raise ValueError("Encountered empty string while normalizing numeric value.")

        sanitized = sanitized.replace(",", "").replace(" ", "")
        sanitized = sanitized.replace("%", "")

        try:
            return int(sanitized)
        except ValueError:
            pass

        try:
            return int(Decimal(sanitized))
        except (InvalidOperation, ValueError):
            match = re.search(r"-?\d+", sanitized)
            if match:
                return int(match.group(0))

        raise ValueError(f"Unable to parse numeric value from '{value}'.")

    async def _maybe_run_numeric_evaluation(
        self, row: dict, question: str, predicted_answer: str, gold_answer: str | int | float, question_id: str | None
    ) -> dict | None:
        if not self._should_use_numeric_eval(row):
            return None
        if gold_answer is None:
            logger.warning(f"Numeric evaluation requested but gold answer missing for question id {question_id}")
            return None

        extracted_value = await self._extract_numeric_answer(question, predicted_answer)
        if extracted_value is None:
            logger.warning(f"Numeric extraction returned no value for question id {question_id}")
            return {
                "correct": 0.0,
                "error": "No numeric value extracted from predicted answer.",
                "extractor_model": self.numeric_eval_model,
                "raw_extracted": None,
            }

        try:
            predicted_numeric = self._normalize_numeric_value(extracted_value)
            gold_numeric = self._normalize_numeric_value(gold_answer)
        except ValueError as exc:
            logger.warning(f"Numeric normalization failed for question id {question_id}: {exc}")
            return {
                "correct": 0.0,
                "error": str(exc),
                "extractor_model": self.numeric_eval_model,
                "raw_extracted": extracted_value,
            }

        absolute_error = abs(gold_numeric - predicted_numeric)
        score = 0.75**absolute_error

        return {
            "correct": score,
            "predicted_numeric": predicted_numeric,
            "gold_numeric": gold_numeric,
            "absolute_error": absolute_error,
            "extractor_model": self.numeric_eval_model,
            "raw_extracted": extracted_value,
            "scoring_rule": "score = 0.75 ** |gold - predicted|",
        }

    @staticmethod
    def _extract_tool_score(tool_data) -> float:
        if not isinstance(tool_data, dict):
            return 0.0
        value = tool_data.get("correct")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _compute_oolong_metrics(self, results: list[dict]) -> dict[str, float]:
        soft_total = soft_correct = 0.0
        numeric_total = numeric_correct = 0.0

        for result in results:
            evaluation_tools = result.get("evaluation_tools", {})
            is_numeric = result.get("is_numeric_question", False)

            if is_numeric:
                numeric_total += 1.0
                numeric_score = self._extract_tool_score(evaluation_tools.get("NumericEvaluation"))
                numeric_correct += numeric_score
            else:
                soft_total += 1.0
                soft_score = self._extract_tool_score(evaluation_tools.get(SOFT_EVALUATOR_TOOL_NAME))
                soft_correct += soft_score

        soft_accuracy = (soft_correct / soft_total) if soft_total else 0.0
        numeric_accuracy = (numeric_correct / numeric_total) if numeric_total else 0.0
        total_questions = soft_total + numeric_total
        combined_accuracy = ((soft_correct + numeric_correct) / total_questions) if total_questions else 0.0

        return {
            "soft_accuracy": soft_accuracy,
            "numeric_accuracy": numeric_accuracy,
            "combined_accuracy": combined_accuracy,
            "soft_total": int(soft_total),
            "numeric_total": int(numeric_total),
            "soft_correct": soft_correct,
            "numeric_correct": numeric_correct,
            "total_questions": int(total_questions),
            "combined_correct": soft_correct + numeric_correct,
        }

    async def _run_row(self, row: dict, system: System, all_metadata: list) -> dict:
        question = row.get("question")
        if question is None:
            raise KeyError("Each row in the benchmark must contain a 'question' entry.")

        context_text = row.get(self.context_text_field)
        if context_text is None:
            raise KeyError(
                f"Context field '{self.context_text_field}' not found in row. "
                "Provide 'context_text_field' in config if the dataset uses a different key."
            )

        document_name = str(
            row.get("context_window_id") or row.get("id") or row.get("question_id") or f"Dataset: {row.get('dataset')}"
        )

        document = await Document.from_plain_text(
            context_text,
            description=self.description,
            document_name=document_name,
            **self.config.get("document_config", {}),
        )

        question = row["context_window_text"].split("\n\nDate:")[0]
        question += f"\n\nThe question is: {row.get('question')}."
        question += "The date, user id and instance text or messages text could be duplicate, but they should be considered as unique. Do not try to remove any instance when getting the answer."

        try:
            all_documents = await contextualize_document_metadata([document], question, model=self.config.get("document_config", {}).get("description_model", "gpt-4.1-mini"))
        except Exception:
            logger.error(f"Error contextualizing document metadata for question: {question}")
            all_documents = [document]

        question_id = row.get("id")
        try:
            answer, metadata = await system.run(question, all_documents, question_id=question_id)
            metadata["gold_answer"] = row.get("answer")
            metadata["predicted_answer"] = answer
        except Exception as e:
            logger.error(f"Error running system for question: {question}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error(e)
            all_metadata.append(
                {
                    "question": question,
                    "error": str(e),
                    "answer": None,
                    "metadata": None,
                    "question_id": question_id,
                }
            )
            return {"error": str(e), "question_id": question_id}

        if metadata is None:
            metadata = {}

        metadata["id"] = question_id
        metadata["context_len"] = row.get("context_len")
        metadata["context_window_id"] = row.get("context_window_id")
        metadata["dataset"] = row.get("dataset")
        if "evidence" in row:
            metadata["evidence"] = row["evidence"]

        numeric_question = self._should_use_numeric_eval(row)

        # Check if we have both pre-merge and post-merge answers
        if "pre_merge_answer" in metadata and "post_merge_answer" in metadata:
            # Evaluate pre-merge answer
            pre_merge_result = await self.evaluator.evaluate(
                question_id=question_id,
                question=question,
                gold_answer=row.get("answer"),
                predicted_answer=metadata["pre_merge_answer"],
            )
            pre_merge_result["is_numeric_question"] = numeric_question

            pre_merge_numeric_result = await self._maybe_run_numeric_evaluation(
                row=row,
                question=question,
                predicted_answer=metadata["pre_merge_answer"],
                gold_answer=row.get("answer"),
                question_id=question_id,
            )
            if pre_merge_numeric_result is not None:
                pre_merge_result.setdefault("evaluation_tools", {})["NumericEvaluation"] = pre_merge_numeric_result

            # Evaluate post-merge answer
            post_merge_result = await self.evaluator.evaluate(
                question_id=question_id,
                question=question,
                gold_answer=row.get("answer"),
                predicted_answer=metadata["post_merge_answer"],
            )
            post_merge_result["is_numeric_question"] = numeric_question

            post_merge_numeric_result = await self._maybe_run_numeric_evaluation(
                row=row,
                question=question,
                predicted_answer=metadata["post_merge_answer"],
                gold_answer=row.get("answer"),
                question_id=question_id,
            )
            if post_merge_numeric_result is not None:
                post_merge_result.setdefault("evaluation_tools", {})["NumericEvaluation"] = post_merge_numeric_result
                if isinstance(metadata, dict) and "predicted_numeric" in post_merge_numeric_result:
                    metadata["predicted_numeric_answer"] = post_merge_numeric_result["predicted_numeric"]

            metadata["is_numeric_question"] = numeric_question

            # Store both evaluations
            result = {
                "question_id": question_id,
                "question": question,
                "gold_answer": row.get("answer"),
                "is_numeric_question": numeric_question,
                "pre_merge_evaluation": pre_merge_result,
                "post_merge_evaluation": post_merge_result,
            }
        # Check if we have both regular and inspect answers (sql_inspect strategy)
        elif "regular_answer" in metadata and "inspect_answer" in metadata:
            # Evaluate regular answer (without inspect_answer)
            regular_result = await self.evaluator.evaluate(
                question_id=question_id,
                question=question,
                gold_answer=row.get("answer"),
                predicted_answer=metadata["regular_answer"],
            )
            regular_result["is_numeric_question"] = numeric_question

            regular_numeric_result = await self._maybe_run_numeric_evaluation(
                row=row,
                question=question,
                predicted_answer=metadata["regular_answer"],
                gold_answer=row.get("answer"),
                question_id=question_id,
            )
            if regular_numeric_result is not None:
                regular_result.setdefault("evaluation_tools", {})["NumericEvaluation"] = regular_numeric_result

            # Evaluate inspect answer
            inspect_result = await self.evaluator.evaluate(
                question_id=question_id,
                question=question,
                gold_answer=row.get("answer"),
                predicted_answer=metadata["inspect_answer"],
            )
            inspect_result["is_numeric_question"] = numeric_question

            inspect_numeric_result = await self._maybe_run_numeric_evaluation(
                row=row,
                question=question,
                predicted_answer=metadata["inspect_answer"],
                gold_answer=row.get("answer"),
                question_id=question_id,
            )
            if inspect_numeric_result is not None:
                inspect_result.setdefault("evaluation_tools", {})["NumericEvaluation"] = inspect_numeric_result
                if isinstance(metadata, dict) and "predicted_numeric" in inspect_numeric_result:
                    metadata["predicted_numeric_answer"] = inspect_numeric_result["predicted_numeric"]

            metadata["is_numeric_question"] = numeric_question

            # Store both evaluations
            result = {
                "question_id": question_id,
                "question": question,
                "gold_answer": row.get("answer"),
                "is_numeric_question": numeric_question,
                "regular_evaluation": regular_result,
                "inspect_evaluation": inspect_result,
            }
        else:
            # Single evaluation
            result = await self.evaluator.evaluate(
                question_id=question_id,
                question=question,
                gold_answer=row.get("answer"),
                predicted_answer=answer,
            )
            result["is_numeric_question"] = numeric_question
            metadata["is_numeric_question"] = numeric_question

            numeric_result = await self._maybe_run_numeric_evaluation(
                row=row,
                question=question,
                predicted_answer=answer,
                gold_answer=row.get("answer"),
                question_id=question_id,
            )
            if numeric_result is not None:
                result.setdefault("evaluation_tools", {})["NumericEvaluation"] = numeric_result
                if isinstance(metadata, dict) and "predicted_numeric" in numeric_result:
                    metadata["predicted_numeric_answer"] = numeric_result["predicted_numeric"]

        all_metadata.append(metadata)
        return result

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
        **kwargs,
    ) -> dict:
        results = []
        dataset = self.dataset
        if self._question_ids:
            dataset = dataset.filter_by_specific_ids(self._question_ids)
            if len(dataset) == 0:
                logger.warning(f"No entries found for question ids: {self._question_ids}")
            else:
                joined_ids = ", ".join(self._question_ids)
                logger.info(f"Filtered dataset to {len(dataset)} item(s) using question ids: {joined_ids}")
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            dataset = dataset.sample(sample_size, random_state=random_state)
        all_metadata = []

        dataset_size = len(dataset)

        if parallel:
            tasks = [self._run_row(row, system, all_metadata) for row in dataset]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        else:
            for idx, row in enumerate(tqdm(dataset, desc="Running experiment")):
                question_id = row.get("id")
                logger.info(
                    f"===============================================\n{idx + 1} of {dataset_size} | Question {question_id}\n==============================================="
                )

                result = await self._run_row(row, system, all_metadata)
                results.append(result)

                # Log both evaluations if we have regular/inspect answers
                if "regular_evaluation" in result:
                    logger.info("=" * 80)
                    logger.info("REGULAR EVALUATION (without inspect_answer):")
                    log_oolong_results(result["regular_evaluation"])
                    logger.info("=" * 80)
                    logger.info("INSPECT EVALUATION (with inspect_answer):")
                    log_oolong_results(result["inspect_evaluation"])
                    logger.info("=" * 80)
                # Log both evaluations if we have pre/post-merge
                elif "pre_merge_evaluation" in result:
                    logger.info("=" * 80)
                    logger.info("PRE-MERGE EVALUATION:")
                    log_oolong_results(result["pre_merge_evaluation"])
                    logger.info("=" * 80)
                    logger.info("POST-MERGE EVALUATION:")
                    log_oolong_results(result["post_merge_evaluation"])
                    logger.info("=" * 80)
                else:
                    log_oolong_results(result)

                # Log current evaluation state and accuracies
                first_result_with_tools = next(
                    (
                        r
                        for r in results
                        if r.get("evaluation_tools") or r.get("pre_merge_evaluation") or r.get("regular_evaluation")
                    ),
                    None,
                )
                if first_result_with_tools:
                    # Check if we have regular/inspect evaluations (sql_inspect strategy)
                    if "regular_evaluation" in first_result_with_tools:
                        # Calculate separate accuracies for regular and inspect
                        regular_accuracies = {}
                        inspect_accuracies = {}

                        for eval_tool, tool_data in first_result_with_tools["regular_evaluation"][
                            "evaluation_tools"
                        ].items():
                            if not isinstance(tool_data, dict) or "correct" not in tool_data:
                                continue

                            # Regular
                            if isinstance(tool_data["correct"], bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("regular_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                regular_accuracies[eval_tool] = accuracy
                            elif isinstance(tool_data["correct"], (int, float)):
                                correct_count = sum(
                                    r.get("regular_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                regular_accuracies[eval_tool] = accuracy

                            # Inspect
                            inspect_tool_data = first_result_with_tools["inspect_evaluation"]["evaluation_tools"].get(
                                eval_tool, {}
                            )
                            if isinstance(inspect_tool_data.get("correct"), bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("inspect_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                inspect_accuracies[eval_tool] = accuracy
                            elif isinstance(inspect_tool_data.get("correct"), (int, float)):
                                correct_count = sum(
                                    r.get("inspect_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                inspect_accuracies[eval_tool] = accuracy

                        logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                        logger.info("REGULAR ACCURACIES (without inspect_answer):")
                        for eval_tool, accuracy in regular_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")

                        # Compute oolong metrics for regular
                        regular_results_for_metrics = [
                            r.get("regular_evaluation", {}) for r in results if "regular_evaluation" in r
                        ]
                        regular_oolong_metrics = self._compute_oolong_metrics(regular_results_for_metrics)
                        logger.info(
                            f"  OolongEvaluation -> soft: {regular_oolong_metrics['soft_accuracy']:.3f} (N={regular_oolong_metrics['soft_total']}), "
                            f"numeric: {regular_oolong_metrics['numeric_accuracy']:.3f} (N={regular_oolong_metrics['numeric_total']}), "
                            f"combined: {regular_oolong_metrics['combined_accuracy']:.3f}"
                        )

                        logger.info("INSPECT ACCURACIES (with inspect_answer):")
                        for eval_tool, accuracy in inspect_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")

                        # Compute oolong metrics for inspect
                        inspect_results_for_metrics = [
                            r.get("inspect_evaluation", {}) for r in results if "inspect_evaluation" in r
                        ]
                        inspect_oolong_metrics = self._compute_oolong_metrics(inspect_results_for_metrics)
                        logger.info(
                            f"  OolongEvaluation -> soft: {inspect_oolong_metrics['soft_accuracy']:.3f} (N={inspect_oolong_metrics['soft_total']}), "
                            f"numeric: {inspect_oolong_metrics['numeric_accuracy']:.3f} (N={inspect_oolong_metrics['numeric_total']}), "
                            f"combined: {inspect_oolong_metrics['combined_accuracy']:.3f}"
                        )
                    # Check if we have pre-merge/post-merge evaluations
                    elif "pre_merge_evaluation" in first_result_with_tools:
                        # Calculate separate accuracies for pre-merge and post-merge
                        pre_merge_accuracies = {}
                        post_merge_accuracies = {}

                        for eval_tool, tool_data in first_result_with_tools["pre_merge_evaluation"][
                            "evaluation_tools"
                        ].items():
                            if not isinstance(tool_data, dict) or "correct" not in tool_data:
                                continue

                            # Pre-merge
                            if isinstance(tool_data["correct"], bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("pre_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                pre_merge_accuracies[eval_tool] = accuracy
                            elif isinstance(tool_data["correct"], (int, float)):
                                correct_count = sum(
                                    r.get("pre_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                pre_merge_accuracies[eval_tool] = accuracy

                            # Post-merge
                            post_tool_data = first_result_with_tools["post_merge_evaluation"]["evaluation_tools"].get(
                                eval_tool, {}
                            )
                            if isinstance(post_tool_data.get("correct"), bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("post_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                post_merge_accuracies[eval_tool] = accuracy
                            elif isinstance(post_tool_data.get("correct"), (int, float)):
                                correct_count = sum(
                                    r.get("post_merge_evaluation", {})
                                    .get("evaluation_tools", {})
                                    .get(eval_tool, {})
                                    .get("correct", 0)
                                    for r in results
                                )
                                accuracy = correct_count / len(results)
                                post_merge_accuracies[eval_tool] = accuracy

                        logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                        logger.info("PRE-MERGE ACCURACIES:")
                        for eval_tool, accuracy in pre_merge_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")

                        # Compute oolong metrics for pre-merge
                        pre_merge_results_for_metrics = [
                            r.get("pre_merge_evaluation", {}) for r in results if "pre_merge_evaluation" in r
                        ]
                        pre_oolong_metrics = self._compute_oolong_metrics(pre_merge_results_for_metrics)
                        logger.info(
                            f"  OolongEvaluation -> soft: {pre_oolong_metrics['soft_accuracy']:.3f} (N={pre_oolong_metrics['soft_total']}), "
                            f"numeric: {pre_oolong_metrics['numeric_accuracy']:.3f} (N={pre_oolong_metrics['numeric_total']}), "
                            f"combined: {pre_oolong_metrics['combined_accuracy']:.3f}"
                        )

                        logger.info("POST-MERGE ACCURACIES:")
                        for eval_tool, accuracy in post_merge_accuracies.items():
                            logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")

                        # Compute oolong metrics for post-merge
                        post_merge_results_for_metrics = [
                            r.get("post_merge_evaluation", {}) for r in results if "post_merge_evaluation" in r
                        ]
                        post_oolong_metrics = self._compute_oolong_metrics(post_merge_results_for_metrics)
                        logger.info(
                            f"  OolongEvaluation -> soft: {post_oolong_metrics['soft_accuracy']:.3f} (N={post_oolong_metrics['soft_total']}), "
                            f"numeric: {post_oolong_metrics['numeric_accuracy']:.3f} (N={post_oolong_metrics['numeric_total']}), "
                            f"combined: {post_oolong_metrics['combined_accuracy']:.3f}"
                        )
                    else:
                        # Single evaluation path
                        current_accuracies = {}
                        for eval_tool, tool_data in first_result_with_tools["evaluation_tools"].items():
                            if not isinstance(tool_data, dict) or "correct" not in tool_data:
                                continue
                            if isinstance(tool_data["correct"], bool):
                                correct_count = sum(
                                    1
                                    for r in results
                                    if r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", False)
                                )
                                accuracy = correct_count / len(results)
                                current_accuracies[eval_tool] = accuracy
                            elif isinstance(tool_data["correct"], (int, float)):
                                correct_count = sum(
                                    r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", 0) for r in results
                                )
                                accuracy = correct_count / len(results)
                                current_accuracies[eval_tool] = accuracy

                        logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                        for eval_tool, accuracy in current_accuracies.items():
                            logger.info(f"{eval_tool} accuracy: {accuracy:.3f}")
                        oolong_metrics = self._compute_oolong_metrics(results)
                        logger.info(
                            f"OolongEvaluation -> soft: {oolong_metrics['soft_accuracy']:.3f} (N={oolong_metrics['soft_total']}), "
                            f"numeric: {oolong_metrics['numeric_accuracy']:.3f} (N={oolong_metrics['numeric_total']}), "
                            f"combined: {oolong_metrics['combined_accuracy']:.3f}"
                        )

                # Log progress and any errors
                if "error" in result:
                    logger.warning(f"Question {question_id} had an error: {result['error']}")
                logger.info(f"Completed {len(results)}/{dataset_size} questions")

        for i, result in enumerate(results):
            result["id"] = all_metadata[i].get("id")
            result["evidence"] = all_metadata[i].get("evidence")

        # Check if we have regular/inspect evaluations (sql_inspect strategy)
        has_inspect_evaluation = len(results) > 0 and "regular_evaluation" in results[0]
        # Check if we have pre-merge/post-merge evaluations
        has_split_evaluation = len(results) > 0 and "pre_merge_evaluation" in results[0]

        if has_inspect_evaluation:
            # Calculate separate summaries for regular and inspect
            regular_summary = {}
            inspect_summary = {}

            for result in results:
                if "regular_evaluation" in result:
                    for tool_name, tool_data in result["regular_evaluation"].get("evaluation_tools", {}).items():
                        if not isinstance(tool_data, dict) or "correct" not in tool_data:
                            continue

                        correct_value = tool_data["correct"]
                        if tool_name not in regular_summary:
                            regular_summary[tool_name] = {"correct": 0.0, "total": 0.0}

                        if isinstance(correct_value, bool):
                            regular_summary[tool_name]["correct"] += 1.0 if correct_value else 0.0
                            regular_summary[tool_name]["total"] += 1.0
                        elif isinstance(correct_value, (int, float)):
                            regular_summary[tool_name]["correct"] += float(correct_value)
                            regular_summary[tool_name]["total"] += 1.0

                if "inspect_evaluation" in result:
                    for tool_name, tool_data in result["inspect_evaluation"].get("evaluation_tools", {}).items():
                        if not isinstance(tool_data, dict) or "correct" not in tool_data:
                            continue

                        correct_value = tool_data["correct"]
                        if tool_name not in inspect_summary:
                            inspect_summary[tool_name] = {"correct": 0.0, "total": 0.0}

                        if isinstance(correct_value, bool):
                            inspect_summary[tool_name]["correct"] += 1.0 if correct_value else 0.0
                            inspect_summary[tool_name]["total"] += 1.0
                        elif isinstance(correct_value, (int, float)):
                            inspect_summary[tool_name]["correct"] += float(correct_value)
                            inspect_summary[tool_name]["total"] += 1.0

            # Compute accuracy per tool
            for tool_name, agg in regular_summary.items():
                total = agg.get("total", 0.0)
                correct = agg.get("correct", 0.0)
                agg["accuracy"] = (correct / total) if total else 0.0

            for tool_name, agg in inspect_summary.items():
                total = agg.get("total", 0.0)
                correct = agg.get("correct", 0.0)
                agg["accuracy"] = (correct / total) if total else 0.0

            # Compute oolong metrics
            regular_results_for_metrics = [
                r.get("regular_evaluation", {}) for r in results if "regular_evaluation" in r
            ]
            regular_oolong_metrics = self._compute_oolong_metrics(regular_results_for_metrics)
            regular_summary["OolongEvaluation"] = {
                "soft_accuracy": regular_oolong_metrics["soft_accuracy"],
                "numeric_accuracy": regular_oolong_metrics["numeric_accuracy"],
                "combined_accuracy": regular_oolong_metrics["combined_accuracy"],
                "soft_questions": regular_oolong_metrics["soft_total"],
                "numeric_questions": regular_oolong_metrics["numeric_total"],
                "accuracy": regular_oolong_metrics["combined_accuracy"],
                "total": regular_oolong_metrics["total_questions"],
                "correct": regular_oolong_metrics["combined_correct"],
            }

            inspect_results_for_metrics = [
                r.get("inspect_evaluation", {}) for r in results if "inspect_evaluation" in r
            ]
            inspect_oolong_metrics = self._compute_oolong_metrics(inspect_results_for_metrics)
            inspect_summary["OolongEvaluation"] = {
                "soft_accuracy": inspect_oolong_metrics["soft_accuracy"],
                "numeric_accuracy": inspect_oolong_metrics["numeric_accuracy"],
                "combined_accuracy": inspect_oolong_metrics["combined_accuracy"],
                "soft_questions": inspect_oolong_metrics["soft_total"],
                "numeric_questions": inspect_oolong_metrics["numeric_total"],
                "accuracy": inspect_oolong_metrics["combined_accuracy"],
                "total": inspect_oolong_metrics["total_questions"],
                "correct": inspect_oolong_metrics["combined_correct"],
            }

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")
            logger.info("")
            logger.info("REGULAR RESULTS SUMMARY (without inspect_answer):")
            logger.info(
                f"  OolongEvaluation -> soft: {regular_oolong_metrics['soft_accuracy']:.3f} (N={regular_oolong_metrics['soft_total']}), "
                f"numeric: {regular_oolong_metrics['numeric_accuracy']:.3f} (N={regular_oolong_metrics['numeric_total']}), "
                f"combined: {regular_oolong_metrics['combined_accuracy']:.3f}"
            )
            logger.info("")
            logger.info("INSPECT RESULTS SUMMARY (with inspect_answer):")
            logger.info(
                f"  OolongEvaluation -> soft: {inspect_oolong_metrics['soft_accuracy']:.3f} (N={inspect_oolong_metrics['soft_total']}), "
                f"numeric: {inspect_oolong_metrics['numeric_accuracy']:.3f} (N={inspect_oolong_metrics['numeric_total']}), "
                f"combined: {inspect_oolong_metrics['combined_accuracy']:.3f}"
            )

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": results,
                "all_metadata": all_metadata,
                "regular_summary": regular_summary,
                "inspect_summary": inspect_summary,
            }
        elif has_split_evaluation:
            # Calculate separate summaries for pre-merge and post-merge
            pre_merge_summary = {}
            post_merge_summary = {}

            for result in results:
                if "pre_merge_evaluation" in result:
                    for tool_name, tool_data in result["pre_merge_evaluation"].get("evaluation_tools", {}).items():
                        if not isinstance(tool_data, dict) or "correct" not in tool_data:
                            continue

                        correct_value = tool_data["correct"]
                        if tool_name not in pre_merge_summary:
                            pre_merge_summary[tool_name] = {"correct": 0.0, "total": 0.0}

                        if isinstance(correct_value, bool):
                            pre_merge_summary[tool_name]["correct"] += 1.0 if correct_value else 0.0
                            pre_merge_summary[tool_name]["total"] += 1.0
                        elif isinstance(correct_value, (int, float)):
                            pre_merge_summary[tool_name]["correct"] += float(correct_value)
                            pre_merge_summary[tool_name]["total"] += 1.0

                if "post_merge_evaluation" in result:
                    for tool_name, tool_data in result["post_merge_evaluation"].get("evaluation_tools", {}).items():
                        if not isinstance(tool_data, dict) or "correct" not in tool_data:
                            continue

                        correct_value = tool_data["correct"]
                        if tool_name not in post_merge_summary:
                            post_merge_summary[tool_name] = {"correct": 0.0, "total": 0.0}

                        if isinstance(correct_value, bool):
                            post_merge_summary[tool_name]["correct"] += 1.0 if correct_value else 0.0
                            post_merge_summary[tool_name]["total"] += 1.0
                        elif isinstance(correct_value, (int, float)):
                            post_merge_summary[tool_name]["correct"] += float(correct_value)
                            post_merge_summary[tool_name]["total"] += 1.0

            # Compute accuracy per tool
            for tool_name, agg in pre_merge_summary.items():
                total = agg.get("total", 0.0)
                correct = agg.get("correct", 0.0)
                agg["accuracy"] = (correct / total) if total else 0.0

            for tool_name, agg in post_merge_summary.items():
                total = agg.get("total", 0.0)
                correct = agg.get("correct", 0.0)
                agg["accuracy"] = (correct / total) if total else 0.0

            # Compute oolong metrics
            pre_merge_results_for_metrics = [
                r.get("pre_merge_evaluation", {}) for r in results if "pre_merge_evaluation" in r
            ]
            pre_oolong_metrics = self._compute_oolong_metrics(pre_merge_results_for_metrics)
            pre_merge_summary["OolongEvaluation"] = {
                "soft_accuracy": pre_oolong_metrics["soft_accuracy"],
                "numeric_accuracy": pre_oolong_metrics["numeric_accuracy"],
                "combined_accuracy": pre_oolong_metrics["combined_accuracy"],
                "soft_questions": pre_oolong_metrics["soft_total"],
                "numeric_questions": pre_oolong_metrics["numeric_total"],
                "accuracy": pre_oolong_metrics["combined_accuracy"],
                "total": pre_oolong_metrics["total_questions"],
                "correct": pre_oolong_metrics["combined_correct"],
            }

            post_merge_results_for_metrics = [
                r.get("post_merge_evaluation", {}) for r in results if "post_merge_evaluation" in r
            ]
            post_oolong_metrics = self._compute_oolong_metrics(post_merge_results_for_metrics)
            post_merge_summary["OolongEvaluation"] = {
                "soft_accuracy": post_oolong_metrics["soft_accuracy"],
                "numeric_accuracy": post_oolong_metrics["numeric_accuracy"],
                "combined_accuracy": post_oolong_metrics["combined_accuracy"],
                "soft_questions": post_oolong_metrics["soft_total"],
                "numeric_questions": post_oolong_metrics["numeric_total"],
                "accuracy": post_oolong_metrics["combined_accuracy"],
                "total": post_oolong_metrics["total_questions"],
                "correct": post_oolong_metrics["combined_correct"],
            }

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")
            logger.info("")
            logger.info("PRE-MERGE RESULTS SUMMARY:")
            logger.info(
                f"  OolongEvaluation -> soft: {pre_oolong_metrics['soft_accuracy']:.3f} (N={pre_oolong_metrics['soft_total']}), "
                f"numeric: {pre_oolong_metrics['numeric_accuracy']:.3f} (N={pre_oolong_metrics['numeric_total']}), "
                f"combined: {pre_oolong_metrics['combined_accuracy']:.3f}"
            )
            logger.info("")
            logger.info("POST-MERGE RESULTS SUMMARY:")
            logger.info(
                f"  OolongEvaluation -> soft: {post_oolong_metrics['soft_accuracy']:.3f} (N={post_oolong_metrics['soft_total']}), "
                f"numeric: {post_oolong_metrics['numeric_accuracy']:.3f} (N={post_oolong_metrics['numeric_total']}), "
                f"combined: {post_oolong_metrics['combined_accuracy']:.3f}"
            )

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": results,
                "all_metadata": all_metadata,
                "pre_merge_summary": pre_merge_summary,
                "post_merge_summary": post_merge_summary,
            }
        else:
            # Original single evaluation path
            results_summary = {}
            for result in results:
                for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                    if not isinstance(tool_data, dict):
                        continue
                    if "correct" not in tool_data:
                        continue

                    correct_value = tool_data["correct"]
                    if tool_name not in results_summary:
                        results_summary[tool_name] = {"correct": 0.0, "total": 0.0}

                    if isinstance(correct_value, bool):
                        results_summary[tool_name]["correct"] += 1.0 if correct_value else 0.0
                        results_summary[tool_name]["total"] += 1.0
                    elif isinstance(correct_value, (int, float)):
                        results_summary[tool_name]["correct"] += float(correct_value)
                        results_summary[tool_name]["total"] += 1.0
                    else:
                        continue

            # compute accuracy per tool
            for tool_name, agg in results_summary.items():
                total = agg.get("total", 0.0)
                correct = agg.get("correct", 0.0)
                agg["accuracy"] = (correct / total) if total else 0.0

            oolong_metrics = self._compute_oolong_metrics(results)
            results_summary["OolongEvaluation"] = {
                "soft_accuracy": oolong_metrics["soft_accuracy"],
                "numeric_accuracy": oolong_metrics["numeric_accuracy"],
                "combined_accuracy": oolong_metrics["combined_accuracy"],
                "soft_questions": oolong_metrics["soft_total"],
                "numeric_questions": oolong_metrics["numeric_total"],
                "accuracy": oolong_metrics["combined_accuracy"],
                "total": oolong_metrics["total_questions"],
                "correct": oolong_metrics["combined_correct"],
            }

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")
            logger.info(
                f"OolongEvaluation -> soft accuracy (N={oolong_metrics['soft_total']}): {oolong_metrics['soft_accuracy']:.3f} | "
                f"numeric accuracy (N={oolong_metrics['numeric_total']}): {oolong_metrics['numeric_accuracy']:.3f} | "
                f"combined: {oolong_metrics['combined_accuracy']:.3f}"
            )

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": results,
                "all_metadata": all_metadata,
                "results_summary": results_summary,
            }
