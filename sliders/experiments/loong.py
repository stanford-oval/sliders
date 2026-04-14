import glob
import json
import os
import re
from typing import Callable

import numpy as np
from tqdm import tqdm

from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document, contextualize_document_metadata
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.globals import SlidersGlobal
from sliders.llm_models import Evaluation, EvaluationScore
from sliders.log_utils import logger
from tqdm.asyncio import tqdm as tqdm_asyncio

file_handle_cache = {}


def create_document_mapping(
    doc_list: list[str], position_format: str = "chinese"
) -> tuple[dict[str, str], dict[str, str]]:
    """Create bidirectional mapping between document titles and positional references.

    Args:
        doc_list: List of document names/titles in order (e.g., row["doc"])
        position_format: Format for positional references:
            - "chinese": Uses 《判决文书N》 format (default, matches Loong legal ground truth)
            - "english": Uses document_N format

    Returns:
        Tuple of (title_to_position, position_to_title) dictionaries.

    Example:
        >>> doc_list = ["case_abc", "case_def", "case_ghi"]
        >>> title_to_pos, pos_to_title = create_document_mapping(doc_list, "chinese")
        >>> title_to_pos["case_abc"]
        '《判决文书1》'
        >>> pos_to_title["《判决文书2》"]
        'case_def'
    """
    title_to_position = {}
    position_to_title = {}

    for idx, doc_name in enumerate(doc_list):
        if position_format == "chinese":
            position = f"《判决文书{idx + 1}》"
        else:
            position = f"document_{idx + 1}"
        title_to_position[doc_name] = position
        position_to_title[position] = doc_name

    return title_to_position, position_to_title


def normalize_answer_to_positions(
    answer: str, doc_list: list[str], case_insensitive: bool = True, position_format: str = "chinese"
) -> str:
    """Replace document titles in an answer with positional references.

    Use this to normalize Sliders output (which uses document titles) to match
    ground truth format (which uses 《判决文书N》 for legal documents).

    Args:
        answer: The answer string containing document titles
        doc_list: List of document names/titles in order (e.g., row["doc"])
        case_insensitive: Whether to perform case-insensitive matching
        position_format: "chinese" for 《判决文书N》, "english" for document_N

    Returns:
        Answer with document titles replaced by positional references

    Example:
        >>> answer = "The verdict is in 和田某某建筑劳务有限公司"
        >>> doc_list = ["和田某某建筑劳务有限公司", "赵某某颜某某民间借贷纠纷"]
        >>> normalize_answer_to_positions(answer, doc_list)
        'The verdict is in 《判决文书1》'
    """
    title_to_position, _ = create_document_mapping(doc_list, position_format)

    result = answer
    # Sort by length descending to avoid partial replacements
    sorted_titles = sorted(title_to_position.keys(), key=len, reverse=True)

    for title in sorted_titles:
        position = title_to_position[title]
        if case_insensitive:
            # Use regex for case-insensitive replacement
            pattern = re.compile(re.escape(title), re.IGNORECASE)
            result = pattern.sub(position, result)
        else:
            result = result.replace(title, position)

    return result


def normalize_answer_to_titles(
    answer: str, doc_list: list[str], case_insensitive: bool = True, position_format: str = "chinese"
) -> str:
    """Replace positional references in an answer with document titles.

    Use this to normalize ground truth (which uses 《判决文书N》 for legal)
    to match Sliders output format (which uses document titles).

    Args:
        answer: The answer string containing positional references
        doc_list: List of document names/titles in order (e.g., row["doc"])
        case_insensitive: Whether to perform case-insensitive matching
        position_format: "chinese" for 《判决文书N》, "english" for document_N

    Returns:
        Answer with positional references replaced by document titles

    Example:
        >>> answer = '{"category": ["《判决文书1》", "《判决文书3》"]}'
        >>> doc_list = ["case_abc", "case_def", "case_ghi"]
        >>> normalize_answer_to_titles(answer, doc_list)
        '{"category": ["case_abc", "case_ghi"]}'
    """
    _, position_to_title = create_document_mapping(doc_list, position_format)

    result = answer

    # Sort by position number descending to avoid 判决文书1 matching in 判决文书10
    def extract_number(pos: str) -> int:
        if position_format == "chinese":
            # Extract number from 《判决文书N》
            match = re.search(r"(\d+)", pos)
            return int(match.group(1)) if match else 0
        else:
            return int(pos.split("_")[1])

    sorted_positions = sorted(
        position_to_title.keys(),
        key=extract_number,
        reverse=True,
    )

    for position in sorted_positions:
        title = position_to_title[position]
        if case_insensitive:
            # Use regex for case-insensitive replacement
            pattern = re.compile(re.escape(position), re.IGNORECASE)
            result = pattern.sub(title, result)
        else:
            result = result.replace(position, title)

    return result


class DocumentReferenceNormalizer:
    """Helper class to normalize document references in answers before evaluation.

    This class provides methods to convert between document titles (used by Sliders)
    and positional references like 《判决文书N》 (used in Loong legal ground truth).

    Usage:
        # Initialize with the document list for this question
        normalizer = DocumentReferenceNormalizer(row["doc"])

        # Option 1: Convert predicted answer (titles) to match ground truth format (positions)
        normalized_prediction = normalizer.to_positions(predicted_answer)

        # Option 2: Convert ground truth (positions) to match predicted answer format (titles)
        normalized_gold = normalizer.to_titles(gold_answer)
    """

    def __init__(self, doc_list: list[str], case_insensitive: bool = True, position_format: str = "chinese"):
        """Initialize the normalizer with a document list.

        Args:
            doc_list: List of document names/titles in order (e.g., row["doc"])
            case_insensitive: Whether to perform case-insensitive matching
            position_format: "chinese" for 《判决文书N》, "english" for document_N
        """
        self.doc_list = doc_list
        self.case_insensitive = case_insensitive
        self.position_format = position_format
        self.title_to_position, self.position_to_title = create_document_mapping(doc_list, position_format)

    def to_positions(self, answer: str) -> str:
        """Convert document titles in an answer to positional references.

        Args:
            answer: The answer string containing document titles

        Returns:
            Answer with document titles replaced by positional references
        """
        return normalize_answer_to_positions(answer, self.doc_list, self.case_insensitive, self.position_format)

    def to_titles(self, answer: str) -> str:
        """Convert positional references in an answer to document titles.

        Args:
            answer: The answer string containing positional references

        Returns:
            Answer with positional references replaced by document titles
        """
        return normalize_answer_to_titles(answer, self.doc_list, self.case_insensitive, self.position_format)

    def get_mapping(self) -> tuple[dict[str, str], dict[str, str]]:
        """Get the bidirectional mapping dictionaries.

        Returns:
            Tuple of (title_to_position, position_to_title) dictionaries
        """
        return self.title_to_position, self.position_to_title


class Loong:
    """Driver for the Loong (Wang et al., 2024) multi-document QA benchmark.

    Covers four domains: English finance, Chinese finance, Chinese legal, and
    English research papers. Each question bundles ~11 documents; answers are
    scored with the official Loong LLM judge.
    """

    def __init__(self, config: dict):
        self.config = config

        benchmark_path = self.config.get("benchmark_path")
        files_dir = self.config.get("files_dir")
        gpt_results_path = self.config.get("gpt_results_path")

        if benchmark_path is None:
            benchmark_path = "/path/to/datasets/loong/loong.jsonl"
        if files_dir is None:
            files_dir = "/path/to/datasets/loong/doc/"
        if gpt_results_path is None:
            gpt_results_path = None

        self.dataset = Dataset(benchmark_path)
        self.dataset = self._apply_filters(self.dataset, config)

        self.files_dir = files_dir
        self.evaluator = Evaluator()

        # Soft evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                eval_class=Evaluation,
                model=self.config["soft_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        # Hard evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                eval_class=Evaluation,
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/loong_evaluator.prompt",
                eval_class=EvaluationScore,
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

    def _apply_filters(self, dataset: Dataset, config: dict) -> Dataset:
        """Apply filters to the dataset based on configuration options."""
        # Priority: specific_ids_csv takes precedence over type/level filters
        if config.get("specific_ids_csv"):
            try:
                import pandas as pd

                id_sample_df = pd.read_csv(config["specific_ids_csv"], comment="#")
                specific_ids = set(id_sample_df["id"].tolist())
                dataset = dataset.filter_by_specific_ids(specific_ids)
                logger.info(f"Filtered dataset by specific IDs: {len(specific_ids)} IDs")
            except Exception as e:
                logger.warning(f"Failed to load specific IDs CSV: {e}, falling back to type/level filters")

        # Apply type filters
        if config.get("filter_by_type"):
            filter_type = config["filter_by_type"]
            dataset = dataset.filter(lambda row: row.get("type") == filter_type)
            logger.info(f"Filtered dataset by type: {filter_type}")

        if config.get("filter_by_types"):
            filter_types = config["filter_by_types"]
            dataset = dataset.filter(lambda row: row.get("type") in filter_types)
            logger.info(f"Filtered dataset by types: {filter_types}")

        # Apply level filters
        if config.get("filter_by_level"):
            filter_level = config["filter_by_level"]
            dataset = dataset.filter(lambda row: row.get("level") == filter_level)
            logger.info(f"Filtered dataset by level: {filter_level}")

        if config.get("filter_by_levels"):
            filter_levels = config["filter_by_levels"]
            dataset = dataset.filter(lambda row: row.get("level") in filter_levels)
            logger.info(f"Filtered dataset by levels: {filter_levels}")

        logger.info(f"Filtered dataset size: {len(dataset)}")
        return dataset

    def description(self, doc_type: str) -> str:
        if doc_type == "paper":
            return "An academic paper from Arxiv."
        elif doc_type == "financial":
            return "A financial document primarily the quarterly and annual reports of a company."
        elif doc_type == "legal":
            return "Consists exclusively of cases adjudicated by the higher and intermediate courts "
        else:
            raise ValueError(f"Unknown document type: {doc_type}")

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
        **kwargs,
    ):
        all_results = []
        dataset = self.dataset
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            logger.info(f"Sampling {sample_size} questions")
            dataset = dataset.sample(sample_size, random_state=random_state)

        # Print dataset statistics
        num_document_counts = {}
        level_counts = {}
        type_counts = {}
        for row in dataset:
            level = row.get("level", "unknown")
            doc_type = row.get("type", "unknown")
            level_counts[level] = level_counts.get(level, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            num_document_counts[len(row["doc"])] = num_document_counts.get(len(row["doc"]), 0) + 1
        dataset_size = len(dataset)
        logger.info(
            f"Dataset level counts: {level_counts}\nDataset type counts: {type_counts}\nTotal dataset size: {dataset_size}"
        )
        logger.info(
            f"No. Documents per question:\navg: {np.mean(list(num_document_counts.keys())):.2f} ± {np.std(list(num_document_counts.keys())):.2f}\nmax: {max(list(num_document_counts.keys()))}\nmin: {min(list(num_document_counts.keys()))}"
        )

        all_results = []
        all_metadata = []
        for row in tqdm(dataset, desc="Running experiment"):
            doc_type = row["type"]
            doc_level = row["level"]

            logger.info(
                f"===============================================\n{len(all_results) + 1} of {dataset_size} | {row['type']}, L{row['level']} | Question {row['id']}\n==============================================="
            )

            logger.info("Loading documents...")
            all_documents = []
            load_document_tasks = []
            for doc_name in row["doc"]:
                file_path = os.path.join(self.files_dir, doc_type)
                if doc_type == "paper":
                    file_path = os.path.join(file_path, doc_name)

                    if not os.path.exists(file_path):
                        return None, {"error": f"Document {file_path} not found"}

                    if self.config.get("docprocesssing", True):
                        document_name = None
                    else:
                        document_name = doc_name
                    load_document_tasks.append(
                        Document.from_markdown(
                            file_path,
                            description=self.description(doc_type),
                            document_name=document_name,
                            **self.config.get("document_config", {}),
                        )
                    )
                elif doc_type == "financial":
                    if str(doc_level) == "4":
                        doc_path = glob.glob(os.path.join(file_path, f"*{doc_name}*.txt"))[0]
                    else:
                        doc_path = glob.glob(os.path.join(file_path, f"*2024-{doc_name}*.txt"))[0]

                    if not os.path.exists(doc_path):
                        return None, {"error": f"Document {doc_path} not found"}

                    json_table_path = os.path.join(
                        self.files_dir, "finance_processed_2", os.path.basename(doc_path) + ".new.tables.json"
                    )

                    if json_table_path and not os.path.exists(json_table_path):
                        logger.error(f"Document {doc_path} not found. JSON table path: {json_table_path}")

                    if self.config.get("docprocesssing", True):
                        document_name = None
                    else:
                        document_name = doc_name

                    load_document_tasks.append(
                        Document.from_file_path(
                            doc_path,
                            description=self.description(doc_type),
                            document_name=document_name,
                            tables_json_path=json_table_path,
                            **self.config.get("document_config", {}),
                        )
                    )
                elif doc_type == "legal":
                    doc_path = os.path.join(file_path, "legal.json")
                    if doc_path in file_handle_cache:
                        legal_js = file_handle_cache[doc_path]
                    else:
                        with open(doc_path, "r") as txt_file:
                            legal_js = json.load(txt_file)
                            file_handle_cache[doc_path] = legal_js

                    if doc_level == 4 and ("阅读以上判决文书，我将给你若干份判决结果：" in row["instruction"]):
                        legal_json_content = legal_js[doc_name]["content"]
                    else:
                        legal_json_content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]

                    load_document_tasks.append(
                        Document.from_plain_text(
                            legal_json_content,
                            description=self.description(doc_type),
                            document_name=doc_name,
                            file_path=doc_path,
                            **self.config.get("document_config", {}),
                        )
                    )
                else:
                    raise ValueError(f"Unknown document type: {doc_type}")

            all_documents = await tqdm_asyncio.gather(*load_document_tasks, desc="Loading documents")

            logger.info("All documents loaded.")

            logger.info("Contextualizing documents...")
            doc_names = [doc.document_name for doc in all_documents]
            question = row["prompt_template"].format(
                docs=doc_names, instruction=row.get("instruction", ""), question=row["question"]
            )
            try:
                if self.config.get("docprocesssing", True):
                    all_documents = await contextualize_document_metadata(all_documents, question, model=self.config.get("document_config", {}).get("description_model", "gpt-4.1-mini"))
                else:
                    # Create a single document by concatenating all chunks
                    concatenated_content = "\n\n".join(
                        [chunk["content"] for doc in all_documents for chunk in doc.chunks]
                    )
                    single_document = Document(
                        content=concatenated_content,
                        processed_content=concatenated_content,
                        tables={},
                        chunks=[{"content": concatenated_content, "chunk_id": 0, "metadata": {}}],
                        document_name="|".join(doc_names),
                        description=self.description(doc_type),
                        file_path=None,
                    )
                    all_documents = [single_document]
                    logger.info(f"Created a single document with {len(single_document.chunks)} chunks")
                    logger.info(f"Length of the document: {len(single_document.content)}")
            except Exception as e:
                logger.error(
                    f"Error contextualizing document metadata: {e}, defaulting to original document descriptions"
                )

            logger.info("Running system...")
            try:
                answer, metadata = await system.run(question, all_documents, question_id=row["id"])
            except Exception as e:
                logger.error(f"Error running system: {e}")

                answer = f"Error running system: {e}"
                metadata = {"error": str(e), "question_id": row["id"]}

            # add question level and type to the metadata dict
            metadata["misc_question_metadata"] = {
                "question_type": row["type"],
                "question_level": row["level"],
                "question_id": row["id"],
            }

            # Normalize document references before evaluation if configured (legal documents only)
            # Options:
            #   "to_positions" - convert predicted answer titles to 《判决文书N》 format
            #   "to_titles"    - convert ground truth 《判决文书N》 to actual document titles
            #   "none" / null  - no normalization (default)
            normalize_mode = self.config.get("normalize_doc_references")
            gold_answer = str(row["answer"]).replace("#", "")
            eval_predicted_answer = answer
            eval_pre_merge_answer = metadata.get("pre_merge_answer")
            eval_post_merge_answer = metadata.get("post_merge_answer")

            # Only apply normalization for legal documents and when a valid mode is specified
            if doc_type == "legal" and normalize_mode and normalize_mode != "none":
                normalizer = DocumentReferenceNormalizer(row["doc"])
                if normalize_mode == "to_positions":
                    # Convert predicted answer (which uses titles) to positions
                    eval_predicted_answer = normalizer.to_positions(answer)
                    if eval_pre_merge_answer:
                        eval_pre_merge_answer = normalizer.to_positions(eval_pre_merge_answer)
                    if eval_post_merge_answer:
                        eval_post_merge_answer = normalizer.to_positions(eval_post_merge_answer)
                    logger.info(f"Normalized predicted answer to positions: {eval_predicted_answer[:200]}...")
                elif normalize_mode == "to_titles":
                    # Convert gold answer (which uses positions) to titles
                    gold_answer = normalizer.to_titles(gold_answer)
                    logger.info(f"Normalized gold answer to titles: {gold_answer[:200]}...")
                else:
                    logger.warning(f"Unknown normalize_doc_references mode: {normalize_mode}")

            # Check if we have both pre-merge and post-merge answers
            if "pre_merge_answer" in metadata and "post_merge_answer" in metadata:
                # Evaluate pre-merge answer
                pre_merge_evaluation = await self.evaluator.evaluate(
                    question_id=row["id"],
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=eval_pre_merge_answer,
                )

                # Evaluate post-merge answer
                post_merge_evaluation = await self.evaluator.evaluate(
                    question_id=row["id"],
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=eval_post_merge_answer,
                )

                # Store both evaluations
                evaluation_result = {
                    "question_id": row["id"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "pre_merge_evaluation": pre_merge_evaluation,
                    "post_merge_evaluation": post_merge_evaluation,
                }

                # Log both results
                logger.info("=" * 80)
                logger.info("PRE-MERGE EVALUATION:")
                log_loong_results(pre_merge_evaluation)
                logger.info("=" * 80)
                logger.info("POST-MERGE EVALUATION:")
                log_loong_results(post_merge_evaluation)
                logger.info("=" * 80)
            else:
                # Single evaluation (no pre-merge/post-merge split)
                evaluation_result = await self.evaluator.evaluate(
                    question_id=row["id"],
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=eval_predicted_answer,
                )
                log_loong_results(evaluation_result)

            all_results.append(evaluation_result)
            all_metadata.append(metadata)

            # Log current evaluation state and accuracies
            if len(all_results) > 0:
                # Check if we have pre-merge/post-merge split evaluations
                if "pre_merge_evaluation" in all_results[0]:
                    # Calculate accuracies for both pre-merge and post-merge
                    pre_merge_accuracies = {}
                    post_merge_accuracies = {}

                    for eval_tool in all_results[0]["pre_merge_evaluation"]["evaluation_tools"].keys():
                        # Pre-merge accuracies
                        if isinstance(
                            all_results[0]["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"], bool
                        ):
                            correct_count = sum(
                                1
                                for result in all_results
                                if "pre_merge_evaluation" in result
                                and result["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"]
                            )
                            accuracy = correct_count / len(all_results)
                            pre_merge_accuracies[eval_tool] = accuracy
                        elif isinstance(
                            all_results[0]["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"],
                            (int, float),
                        ):
                            correct_count = sum(
                                result["pre_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"]
                                for result in all_results
                                if "pre_merge_evaluation" in result
                            )
                            accuracy = correct_count / len(all_results)
                            pre_merge_accuracies[eval_tool] = accuracy

                        # Post-merge accuracies
                        if isinstance(
                            all_results[0]["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"], bool
                        ):
                            correct_count = sum(
                                1
                                for result in all_results
                                if "post_merge_evaluation" in result
                                and result["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"]
                            )
                            accuracy = correct_count / len(all_results)
                            post_merge_accuracies[eval_tool] = accuracy
                        elif isinstance(
                            all_results[0]["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"],
                            (int, float),
                        ):
                            correct_count = sum(
                                result["post_merge_evaluation"]["evaluation_tools"][eval_tool]["correct"]
                                for result in all_results
                                if "post_merge_evaluation" in result
                            )
                            accuracy = correct_count / len(all_results)
                            post_merge_accuracies[eval_tool] = accuracy

                    logger.info(f"=== CURRENT EVALUATION STATE ({len(all_results)}/{dataset_size}) ===")
                    logger.info("PRE-MERGE ACCURACIES:")
                    for eval_tool, accuracy in pre_merge_accuracies.items():
                        logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")
                    logger.info("POST-MERGE ACCURACIES:")
                    for eval_tool, accuracy in post_merge_accuracies.items():
                        logger.info(f"  {eval_tool} accuracy: {accuracy:.3f}")
                else:
                    # Calculate current accuracies for each evaluation tool (single evaluation)
                    current_accuracies = {}
                    for eval_tool in all_results[0]["evaluation_tools"].keys():
                        if isinstance(all_results[0]["evaluation_tools"][eval_tool]["correct"], bool):
                            correct_count = sum(
                                1 for result in all_results if result["evaluation_tools"][eval_tool]["correct"]
                            )
                            accuracy = correct_count / len(all_results)
                            current_accuracies[eval_tool] = accuracy
                        elif isinstance(all_results[0]["evaluation_tools"][eval_tool]["correct"], (int, float)):
                            correct_count = sum(
                                result["evaluation_tools"][eval_tool]["correct"] for result in all_results
                            )
                            accuracy = correct_count / len(all_results)
                            current_accuracies[eval_tool] = accuracy
                        else:
                            raise ValueError(
                                f"Unknown evaluation tool type: {type(all_results[0]['evaluation_tools'][eval_tool]['correct'])}"
                            )
                    logger.info(f"=== CURRENT EVALUATION STATE ({len(all_results)}/{dataset_size}) ===")
                    for eval_tool, accuracy in current_accuracies.items():
                        logger.info(f"{eval_tool} accuracy: {accuracy:.3f}")

            # Log progress and any errors
            if "error" in metadata:
                logger.warning(f"Question {row['id']} had an error: {metadata['error']}")
            logger.info(f"Completed {len(all_results)}/{dataset_size} questions")

        # results summary
        results_summary = {}
        pre_merge_summary = {}
        post_merge_summary = {}

        # Check if we have pre-merge/post-merge evaluations
        has_split_evaluation = len(all_results) > 0 and "pre_merge_evaluation" in all_results[0]

        if has_split_evaluation:
            # Calculate separate summaries for pre-merge and post-merge
            for result in all_results:
                if "pre_merge_evaluation" in result:
                    for tool_name, tool_data in result["pre_merge_evaluation"].get("evaluation_tools", {}).items():
                        if tool_name not in pre_merge_summary:
                            pre_merge_summary[tool_name] = {"correct": 0, "total": 0}

                        is_correct = False
                        if isinstance(tool_data, dict):
                            is_correct = bool(tool_data.get("correct", False))

                        pre_merge_summary[tool_name]["correct"] += int(is_correct)
                        pre_merge_summary[tool_name]["total"] += 1

                if "post_merge_evaluation" in result:
                    for tool_name, tool_data in result["post_merge_evaluation"].get("evaluation_tools", {}).items():
                        if tool_name not in post_merge_summary:
                            post_merge_summary[tool_name] = {"correct": 0, "total": 0}

                        is_correct = False
                        if isinstance(tool_data, dict):
                            is_correct = bool(tool_data.get("correct", False))

                        post_merge_summary[tool_name]["correct"] += int(is_correct)
                        post_merge_summary[tool_name]["total"] += 1

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(all_results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")
            logger.info("")
            logger.info("PRE-MERGE RESULTS SUMMARY:")
            for tool_name, stats in pre_merge_summary.items():
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                logger.info(f"  {tool_name}: {stats['correct']}/{stats['total']} ({accuracy:.3f})")
            logger.info("")
            logger.info("POST-MERGE RESULTS SUMMARY:")
            for tool_name, stats in post_merge_summary.items():
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                logger.info(f"  {tool_name}: {stats['correct']}/{stats['total']} ({accuracy:.3f})")

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": all_results,
                "all_metadata": all_metadata,
                "pre_merge_summary": pre_merge_summary,
                "post_merge_summary": post_merge_summary,
            }
        else:
            # Original single evaluation path
            for result in all_results:
                for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                    if tool_name not in results_summary:
                        results_summary[tool_name] = {"correct": 0, "total": 0}

                    is_correct = False
                    if isinstance(tool_data, dict):
                        is_correct = bool(tool_data.get("correct", False))

                    results_summary[tool_name]["correct"] += int(is_correct)
                    results_summary[tool_name]["total"] += 1

            # Final summary
            successful_count = len([m for m in all_metadata if "error" not in m])
            error_count = len([m for m in all_metadata if "error" in m])
            logger.info("=== EXPERIMENT COMPLETE ===")
            logger.info(f"Total questions processed: {len(all_results)}")
            logger.info(f"Successful runs: {successful_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Expected sample size: {dataset_size}")

            return {
                "experiment_id": SlidersGlobal.experiment_id,
                "results": all_results,
                "all_metadata": all_metadata,
                "results_summary": results_summary,
            }


def log_loong_results(result):
    logger.info(f"Gold Answer: {result['gold_answer']}")
    logger.info(f"Predicted Answer: {result['predicted_answer']}")
    for key, value in result["evaluation_tools"].items():
        if isinstance(value, dict) and "correct" in value:
            logger.info(f"{key}: {value['correct']}")
        else:
            logger.info(f"{key}: {value}")
