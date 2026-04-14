from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
import uuid
from copy import deepcopy

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from sliders.callbacks.logging import LoggingHandler
from sliders.document import Document
from sliders.llm_models import Action, ExtractedTable, SQLAnswer, TableProcessingNeeded
from sliders.llm_tools import DuckSQLBasic, run_sql_query
from sliders.llm import get_llm_client, load_fewshot_prompt_template
from sliders.log_utils import logger
from sliders.modules import ExtractSchema, GenerateSchema, MergedTables, QuestionRephraser
from sliders.utils import prepare_schema_repr, format_table, tables_to_template_dicts
from sliders.modules.answer_techniques import run_inspect_answer

if TYPE_CHECKING:
    from sliders.llm_models import Tables, Table

CURRENT_DIR = Path(__file__).parent


def save_tables_to_dir(
    tables: list[ExtractedTable],
    save_dir: str | Path,
    question_id: str,
    stage: str,
) -> None:
    """
    Save ExtractedTable dataframes to CSV files.

    Args:
        tables: List of ExtractedTable objects to save
        save_dir: Directory to save tables to
        question_id: Question ID for organizing files
        stage: Stage name (e.g., "pre_merge", "post_merge")
    """
    save_path = Path(save_dir) / question_id / stage
    save_path.mkdir(parents=True, exist_ok=True)

    for table in tables:
        if table.dataframe is not None and not table.dataframe.empty:
            table_file = save_path / f"{table.name}.csv"
            table.dataframe.to_csv(table_file, index=False)
            logger.info(f"Saved {stage} table '{table.name}' to {table_file}")


class System(ABC):
    def __init__(self, config):
        self.config = config
        self._setup_chains()

    @abstractmethod
    async def run(self, question: str, documents: list[Document], question_id: str = "") -> tuple[str, dict]:
        pass

    @abstractmethod
    def _setup_chains(self):
        pass


class SlidersAgent(System):
    def __init__(self, config):
        super().__init__(config)
        self._setup_modules()

    def _setup_modules(self):
        extract_config = dict(self.config.get("extract_schema", {}))
        merge_config = dict(self.config.get("merge_tables", {}))
        self.config["extract_schema"] = extract_config
        self.config["merge_tables"] = merge_config

        self.extract_schema = ExtractSchema(extract_config, model_config=self.config["models"])
        self.generate_schema = GenerateSchema(
            self.config.get("generate_schema", {}), model_config=self.config["models"]
        )

        if "include_quotes" not in merge_config:
            merge_config["include_quotes"] = extract_config.get("extract_quotes", True)
        self.merge_tables = MergedTables(merge_config, model_config=self.config["models"])

        rephrase_config_raw = self.config.get("rephrase_question", False)
        self.rephrase_question_enabled = False
        self.question_rephraser = None

        if isinstance(rephrase_config_raw, dict):
            self.rephrase_question_enabled = rephrase_config_raw.get("enabled", True)
            module_config = {k: v for k, v in rephrase_config_raw.items() if k != "enabled"}
        else:
            self.rephrase_question_enabled = bool(rephrase_config_raw)
            module_config = {}

        if self.rephrase_question_enabled:
            self.question_rephraser = QuestionRephraser(module_config, model_config=self.config["models"])

        # Load taxonomy for classification if needed by schema generation or rephrasing
        self.taxonomy = None
        self.classification_chain = None

        # Check if taxonomy is needed
        taxonomy_needed = False
        taxonomy_path = None

        # Check if schema generation needs it
        if self.config.get("generate_schema", {}).get("generate_schema_type") == "library_based":
            taxonomy_path = self.config.get("generate_schema", {}).get("library_of_guidelines_path")
            if taxonomy_path:
                taxonomy_needed = True

        # Check if rephrasing needs it
        if self.rephrase_question_enabled:
            rephrase_taxonomy_path = self.config.get("rephrase_question", {}).get("library_of_guidelines_path")
            if rephrase_taxonomy_path:
                taxonomy_needed = True
                taxonomy_path = rephrase_taxonomy_path

        # Load taxonomy and create classification chain if needed
        if taxonomy_needed and taxonomy_path:
            self.taxonomy = self._load_taxonomy(taxonomy_path)
            # Create classification chain
            if "select_guidelines_for_schema" in self.config["models"]:
                self.classification_chain = self._create_classification_chain()

    def _load_taxonomy(self, taxonomy_path: str) -> dict:
        """Load the taxonomy for classification."""
        full_path = Path(CURRENT_DIR) / taxonomy_path
        with open(full_path, "r") as f:
            return json.load(f)

    def _create_classification_chain(self):
        """Create the classification chain for question/document type classification."""
        from sliders.llm.prompts import load_fewshot_prompt_template

        llm_client = get_llm_client(**self.config["models"]["select_guidelines_for_schema"])
        select_guidelines_template = load_fewshot_prompt_template(
            template_file="sliders/select_guidelines_for_schema.prompt",
            template_blocks=[],
        )
        return select_guidelines_template | llm_client.with_structured_output(method="json_mode")

    async def classify_question_and_document(
        self, question: str, documents: list[Document], metadata: dict
    ) -> tuple[str, str]:
        """
        Classify the question and document types for use in guideline selection.
        Returns (question_type, document_type).
        """
        if self.taxonomy is None or self.classification_chain is None:
            return "simple", "others"

        handler = LoggingHandler(
            prompt_file="sliders/select_guidelines_for_schema.prompt",
            metadata={
                "question": question,
                "document_descriptions": [doc.description for doc in documents],
                "stage": "classify_question_document",
                "objective": "select_guidelines",
                **(metadata or {}),
            },
        )

        category_descriptions = ""
        category_descriptions += "## Document Type Descriptions\n"
        for category, description in self.taxonomy["document_type"].items():
            category_descriptions += f"- {category}: {description}\n"
        category_descriptions += "## Question Type Descriptions\n"
        for category, description in self.taxonomy["question_type"].items():
            category_descriptions += f"- {category}: {description}\n"

        selected_guidelines = await self.classification_chain.ainvoke(
            {
                "question": question,
                "document_descriptions": [doc.description for doc in documents],
                "category_descriptions": category_descriptions,
            },
            config={"callbacks": [handler]},
        )
        question_type = selected_guidelines.get("question_type", "simple")
        document_type = selected_guidelines.get("document_type", "others")

        # Store classification in metadata
        metadata["schema"]["question_type"] = question_type
        metadata["schema"]["document_type"] = document_type
        metadata["schema"]["classification_reasoning"] = selected_guidelines.get("reasoning", "")

        return question_type, document_type

    def _setup_chains(self):
        # Answer question from schema when we find tables
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file="sliders/answer_from_schema_tool_use.prompt",
            template_blocks=[],
        )
        self.answer_question_chain = answer_template | answer_llm_client.with_structured_output(Action)

        force_answer_llm_client = get_llm_client(**self.config["models"]["force_answer"])
        force_answer_template = load_fewshot_prompt_template(
            template_file="sliders/force_answer_from_schema_tool_use.prompt",
            template_blocks=[],
        )
        self.force_answer_question_chain = force_answer_template | force_answer_llm_client.with_structured_output(
            SQLAnswer
        )

        direct_answer_llm_client = get_llm_client(**self.config["models"]["direct_answer"])
        direct_answer_template = load_fewshot_prompt_template(
            template_file="sliders/direct_answer_from_tables.prompt",
            template_blocks=[],
        )
        self.direct_answer_chain = direct_answer_template | direct_answer_llm_client | StrOutputParser()

        check_if_merge_needed_llm_client = get_llm_client(**self.config["models"]["check_if_merge_needed"])
        check_if_merge_needed_template = load_fewshot_prompt_template(
            template_file="sliders/check_if_merge_needed.prompt",
            template_blocks=[],
        )
        self.check_if_merge_needed_chain = (
            check_if_merge_needed_template
            | check_if_merge_needed_llm_client.with_structured_output(TableProcessingNeeded)
        )

        # Answer question from schema when we don't find any tables
        answer_no_table_llm_client = get_llm_client(**self.config["models"]["answer_no_table"])
        answer_no_table_template = load_fewshot_prompt_template(
            template_file="sliders/answer_from_schema_no_table.prompt",
            template_blocks=[],
        )
        self.answer_question_no_table_chain = (
            answer_no_table_template | answer_no_table_llm_client.with_structured_output(Action)
        )

        # Generate final answer from SQL output
        tool_output_llm_client = get_llm_client(**self.config["models"]["answer_tool_output"])
        tool_output_template = load_fewshot_prompt_template(
            template_file="sliders/answer_with_tool_use_output.prompt",
            template_blocks=[],
        )
        self.tool_output_chain = tool_output_template | tool_output_llm_client | StrOutputParser()

        if self.config.get("generate_task_guidelines", False):
            task_guidelines_llm_client = get_llm_client(**self.config["models"]["task_guidelines"])
            task_guidelines_template = load_fewshot_prompt_template(
                template_file="sliders/task_guidelines.prompt",
                template_blocks=[],
            )
            self.create_task_guidelines_chain = task_guidelines_template | task_guidelines_llm_client

    def _initialize_metadata(
        self, question: str, documents: list[Document], start_time: float, question_id: str = ""
    ) -> dict:
        return {
            # Basic request information
            "question": question,
            "num_documents": len(documents),
            "document_names": [doc.document_name for doc in documents],
            "document_sizes": [len(doc.content) for doc in documents],
            "total_chunks": sum(len(doc.chunks) for doc in documents),
            # Processing pipeline timing
            "timing": {
                "start_time": start_time,
                "schema_generation": {},
                "schema_extraction": {},
                "table_merging": {},
                "answer_generation": {},
                "total_duration": None,
            },
            # Schema and extraction metrics
            "schema": {"generated_classes": 0, "total_fields": 0, "generation_tokens": 0, "generation_time": 0},
            # Extraction statistics
            "extraction": {
                "chunks_processed": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "retry_attempts": 0,
                "extraction_time": 0,
            },
            # Table merging information
            "merging": {
                "tables_created": 0,
                "sql_queries_executed": 0,
                "merge_failures": 0,
                "total_rows_processed": 0,
                "merging_time": 0,
                "merged_tables_dir_path": "",
            },
            # Answer generation metrics
            "answer_generation": {
                "sql_execution_attempts": 0,
                "sql_execution_errors": 0,
                "final_answer_tokens": 0,
                "used_tables": False,
                "answer_time": 0,
            },
            # Quality and performance metrics
            "quality": {"tables_with_data": 0, "empty_tables": 0, "data_completeness_score": 0.0},
            # Error tracking
            "errors": [],
            # Question ID
            "question_id": question_id,
            # Output folder for intermediate artifacts (e.g. debug CSVs)
            "output_folder": self.config.get("output_folder", "."),
        }

    def _finalize_metadata(self, metadata: dict, tables: list["Table"], start_time: float) -> dict:
        metadata["timing"]["total_duration"] = time.time() - start_time
        metadata["quality"]["tables_with_data"] = len(
            [t for t in tables if t.dataframe is not None and not t.dataframe.empty]
        )
        metadata["quality"]["empty_tables"] = len([t for t in tables if t.dataframe is None or t.dataframe.empty])

        # Calculate data completeness score
        if metadata["quality"]["tables_with_data"] + metadata["quality"]["empty_tables"] > 0:
            metadata["quality"]["data_completeness_score"] = metadata["quality"]["tables_with_data"] / (
                metadata["quality"]["tables_with_data"] + metadata["quality"]["empty_tables"]
            )
        return metadata

    async def run(self, question: str, documents: list[Document], question_id: str = "") -> tuple[str, dict]:
        logger.info(f"Running SlidersAgent with question: {question}")
        start_time = time.time()

        # Initialize comprehensive metadata structure
        metadata = self._initialize_metadata(question, documents, start_time, question_id)

        # Generate task guidelines internally
        if self.config.get("generate_task_guidelines", False):
            task_guidelines_handler = LoggingHandler(
                prompt_file="sliders/task_guidelines.prompt",
                metadata={
                    "question": question,
                    "document_descriptions": [doc.description for doc in documents],
                    "question_id": question_id,
                },
            )
            task_guidelines = await self.create_task_guidelines_chain.ainvoke(
                {"question": question, "document_descriptions": [doc.description for doc in documents]},
                config={"callbacks": [task_guidelines_handler]},
            )
        else:
            task_guidelines = None

        original_question = question
        schema_question = original_question
        extraction_question = original_question
        merge_question = original_question
        answer_question = original_question

        # Classify question and document type if needed by schema generation or rephrasing
        question_type = None
        document_type = None
        if self.taxonomy is not None and self.classification_chain is not None:
            logger.info("Classifying question and document type...")
            try:
                question_type, document_type = await self.classify_question_and_document(
                    original_question, documents, metadata
                )
                logger.info(f"Classification - Question type: {question_type}, Document type: {document_type}")
            except Exception as exc:
                logger.exception(f"Classification failed: {exc}")
                metadata["errors"].append(
                    {
                        "stage": "classification",
                        "error": str(exc),
                        "question": original_question,
                        "question_id": question_id,
                    }
                )

        if self.question_rephraser is not None:
            try:
                component_questions = await self.question_rephraser.rephrase(
                    original_question, documents, metadata, document_type=document_type, question_type=question_type
                )
            except Exception as exc:
                logger.exception(f"Question rephrasing failed: {exc}")
                metadata["errors"].append(
                    {
                        "stage": "rephrase_question",
                        "error": str(exc),
                        "question": original_question,
                        "question_id": question_id,
                    }
                )
                metadata["rephrase_question"] = {
                    "enabled": True,
                    "original_question": original_question,
                    "error": str(exc),
                    "questions": None,
                    "fallback_components": ["schema", "extraction", "merge", "answer"],
                    "question_list": [original_question] * 4,
                }
            else:
                if component_questions:
                    schema_question = component_questions.schema_question
                    extraction_question = component_questions.extraction_question
                    merge_question = component_questions.merge_question
                    answer_question = component_questions.answer_question
                    logger.info("Rephrased questions")
                    logger.info(f"Schema question: {schema_question}")
                    logger.info(f"Extraction question: {extraction_question}")
                    logger.info(f"Merge question: {merge_question}")
                    logger.info(f"Answer question: {answer_question}")
                    metadata["rephrase_question"] = {
                        "enabled": True,
                        "original_question": original_question,
                        "questions": component_questions.as_dict(),
                        "fallback_components": sorted(component_questions.fallback_components),
                        "question_list": component_questions.as_list(),
                    }
        elif "rephrase_question" not in metadata:
            metadata["rephrase_question"] = {
                "enabled": False,
                "original_question": original_question,
            }

        logger.info("Generating schema...")
        schema = await self.generate_schema.generate(
            question,
            documents,
            metadata,
            task_guidelines,
            question_type=question_type,
            document_type=document_type,
        )

        logger.info("Extracting tables from documents...")
        extracted_tables = await self.extract_schema.extract(
            extraction_question, schema, documents, metadata, task_guidelines
        )

        # Generate pre-merge answer if perform_merge is True
        pre_merge_answer = None
        pre_merge_metadata = None
        if self.config.get("perform_merge", True):
            logger.info("=" * 80)
            logger.info("GENERATING PRE-MERGE ANSWER (WITHOUT MERGING)")
            logger.info("=" * 80)

            # Create pre-merge tables (same as if perform_merge were False)
            pre_merge_tables = []
            for table_name in extracted_tables.keys():
                sid = uuid.uuid4().hex[:8]
                table_data_df, new_table_name = self.merge_tables.create_table_data(
                    table_name, extracted_tables[table_name], sid
                )
                if table_data_df.shape[1] == 0:
                    logger.info(f"Skipping table {table_name} (pre-merge) because no columns were extracted.")
                    continue
                pre_merge_tables.append(
                    ExtractedTable(
                        name=table_name,
                        tables=schema,
                        sql_query=None,
                        dataframe=table_data_df,
                        dataframe_table_name=new_table_name,
                        table_str=format_table(table_data_df),
                    )
                )

            # Save pre-merge tables if configured
            if self.config.get("save_tables", False) and self.config.get("save_tables_dir"):
                save_tables_to_dir(
                    pre_merge_tables,
                    self.config["save_tables_dir"],
                    question_id,
                    "pre_merge",
                )

            # Use deepcopy for pre-merge metadata
            pre_merge_metadata = deepcopy(metadata)
            pre_merge_metadata["answer_stage"] = "pre_merge"

            # Generate pre-merge answer
            if self.config.get("force_sql", False):
                pre_merge_answer = await self._force_answer_question_from_tables(
                    question, pre_merge_tables, schema, pre_merge_metadata
                )
            else:
                pre_merge_answer = await self._answer_question_from_tables(
                    question, pre_merge_tables, schema, pre_merge_metadata
                )

            if isinstance(pre_merge_answer, AIMessage):
                pre_merge_answer = pre_merge_answer.content

            logger.info("=" * 80)
            logger.info("PRE-MERGE ANSWER:")
            logger.info(pre_merge_answer)
            logger.info("=" * 80)

        logger.info(f"Perform merge: {self.config.get('perform_merge', True)}")
        if self.config.get("perform_merge", True):
            merge_needed_dict = {}
            check_if_merge_needed = True
            logger.info(f"Check if merge needed: {self.config.get('check_if_merge_needed', False)}")
            if self.config.get("check_if_merge_needed", False):
                for table_name, table_data in extracted_tables.items():
                    sid = uuid.uuid4().hex[:8]
                    table_data_df, new_table_name = self.merge_tables.create_table_data(
                        table_name, extracted_tables[table_name], sid
                    )
                    if table_data_df.shape[1] == 0:
                        merge_needed_dict[table_name] = False
                        continue
                    formatted_table = format_table(table_data_df)
                    check_if_merge_needed_handler = LoggingHandler(
                        prompt_file="sliders/check_if_merge_needed.prompt",
                        metadata={
                            "question": merge_question,
                            "table": formatted_table,
                            "schema": prepare_schema_repr(schema),
                            "question_id": question_id,
                        },
                    )
                    check_if_merge_needed_output = await self.check_if_merge_needed_chain.ainvoke(
                        {"question": merge_question, "table": formatted_table, "schema": prepare_schema_repr(schema)},
                        config={"callbacks": [check_if_merge_needed_handler]},
                    )
                    check_if_merge_needed = check_if_merge_needed_output.processing_needed
                    merge_needed_dict[table_name] = check_if_merge_needed
                    logger.info(f"Table {table_name} merge needed: {check_if_merge_needed}")
            else:
                for table_name, table_data in extracted_tables.items():
                    merge_needed_dict[table_name] = True

            tables = []
            for table_name, merge_needed in merge_needed_dict.items():
                if merge_needed:
                    # Merge tables from different documents
                    logger.info("Merging tables from different documents...")
                    tables.extend(
                        await self.merge_tables.merge_chunks_tables(
                            {table_name: extracted_tables[table_name]},
                            documents,
                            merge_question,
                            schema,
                            metadata,
                        )
                    )

                    logger.info(f"Merged tables directory path: {metadata['merging']['merged_tables_dir_path']}")
                else:
                    sid = uuid.uuid4().hex[:8]
                    table_data_df, new_table_name = self.merge_tables.create_table_data(
                        table_name, extracted_tables[table_name], sid
                    )
                    if table_data_df.shape[1] == 0:
                        logger.info(f"Skipping table {table_name} because no columns were extracted.")
                        continue
                    # Convert extracted_tables to ExtractedTable objects without merging
                    tables.append(
                        ExtractedTable(
                            name=table_name,
                            tables=schema,
                            sql_query=None,
                            dataframe=table_data_df,
                            dataframe_table_name=new_table_name,
                            table_str=format_table(table_data_df),
                        )
                    )
                    logger.info(f"Created {len(tables)} tables without merging for table {table_name}")
        else:
            tables = []
            for table_name in extracted_tables.keys():
                sid = uuid.uuid4().hex[:8]
                table_data_df, new_table_name = self.merge_tables.create_table_data(
                    table_name, extracted_tables[table_name], sid
                )
                if table_data_df.shape[1] == 0:
                    logger.info(f"Skipping table {table_name} because no columns were extracted.")
                    continue
                tables.append(
                    ExtractedTable(
                        name=table_name,
                        tables=schema,
                        sql_query=None,
                        dataframe=table_data_df,
                        dataframe_table_name=new_table_name,
                        table_str=format_table(table_data_df),
                    )
                )

        # Save post-merge tables if configured
        if self.config.get("save_tables", False) and self.config.get("save_tables_dir"):
            stage = "post_merge" if self.config.get("perform_merge", True) else "no_merge"
            save_tables_to_dir(
                tables,
                self.config["save_tables_dir"],
                question_id,
                stage,
            )

        # Answer question
        if self.config.get("perform_merge", True):
            logger.info("=" * 80)
            logger.info("GENERATING POST-MERGE ANSWER (WITH MERGING)")
            logger.info("=" * 80)
            metadata["answer_stage"] = "post_merge"
        else:
            logger.info("Answering question from tables...")

        if self.config.get("force_sql", False):
            answer = await self._force_answer_question_from_tables(question, tables, schema, metadata)
        else:
            answer = await self._answer_question_from_tables(question, tables, schema, metadata)

        # Log reconciliation stats summary if available
        if "answer_generation" in metadata and "reconciliation_stats_summary" in metadata["answer_generation"]:
            logger.info("=" * 80)
            logger.info("RECONCILIATION STATISTICS SUMMARY")
            logger.info("=" * 80)
            logger.info(metadata["answer_generation"]["reconciliation_stats_summary"])
            logger.info("=" * 80)

        # Log citation paragraph if available
        if "answer_generation" in metadata and "citation_paragraph" in metadata["answer_generation"]:
            citation_paragraph = metadata["answer_generation"]["citation_paragraph"]
            if citation_paragraph is not None:
                logger.info("=" * 80)
                logger.info("CITATION PARAGRAPH")
                logger.info("=" * 80)
                logger.info(citation_paragraph)
                logger.info("=" * 80)

        # Finalize metadata
        metadata = self._finalize_metadata(metadata, tables, start_time)

        if isinstance(answer, AIMessage):
            answer = answer.content

        # Log both answers together if we have pre-merge answer
        if self.config.get("perform_merge", True) and pre_merge_answer is not None:
            logger.info("=" * 80)
            logger.info("PRE-MERGE ANSWER:")
            logger.info(pre_merge_answer)
            logger.info("=" * 80)
            logger.info("POST-MERGE ANSWER:")
            logger.info(answer)
            logger.info("=" * 80)

            # Store both answers in metadata
            metadata["pre_merge_answer"] = pre_merge_answer
            metadata["post_merge_answer"] = answer
            metadata["pre_merge_metadata"] = pre_merge_metadata

        # Finalize metadata
        metadata = self._finalize_metadata(metadata, tables, start_time)

        return answer, metadata

    async def _force_answer_question_from_tables(
        self, question: str, tables: list["Table"], schema: Tables, metadata: dict
    ) -> str:
        answer_start_time = time.time()

        with DuckSQLBasic() as duck_sql_conn:
            filtered_tables = []
            for table in tables:
                df = getattr(table, "dataframe", None)
                if df is None or getattr(df, "shape", (0, 0))[1] == 0:
                    logger.info(f"Skipping table {table.name} due to empty or missing dataframe.")
                    table.table_str = None
                    continue

                duck_sql_conn.register(
                    df,
                    table.dataframe_table_name,
                    schema=schema,
                    schema_table_name=table.name,
                )
                table.table_str = (
                    str(tuple(df.columns.to_list()))
                    + "\n"
                    + "\n".join([str(row) for row in df.to_records(index=False)])
                )
                filtered_tables.append(table)

            tables = filtered_tables

            # If no tables, answer question from schema
            if len(tables) == 0:
                metadata["answer_generation"]["used_tables"] = False
                answer_question_no_table_handler = LoggingHandler(
                    prompt_file="sliders/answer_from_schema_no_table.prompt",
                    metadata={
                        "question": question,
                        "classes": tables,
                        "feedback": None,
                        "question_id": metadata.get("question_id", None),
                        "stage": "answer_question_no_table",
                    },
                )
                result = await self.answer_question_no_table_chain.ainvoke(
                    {"question": question, "classes": tables, "feedback": None},
                    config={"callbacks": [answer_question_no_table_handler]},
                )
                final_answer = result.answer
            # If tables, answer question from tables
            else:
                metadata["answer_generation"]["used_tables"] = True

                async def query_llm_for_sql(feedback: str = None) -> SQLAnswer:
                    force_answer_question_handler = LoggingHandler(
                        prompt_file="sliders/force_answer_from_schema_tool_use.prompt",
                        metadata={
                            "question": question,
                            "classes": tables,
                            "feedback": feedback,
                            "question_id": metadata.get("question_id", None),
                            "stage": "generate_sql_answer",
                        },
                    )
                    result = await self.force_answer_question_chain.ainvoke(
                        {
                            "tables": tables_to_template_dicts(tables),
                            "question": question,
                            "feedback": feedback,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [force_answer_question_handler]},
                    )

                    tool_output, error = run_sql_query(result.sql_query, duck_sql_conn)

                    if error:
                        logger.error(f"Error running SQL query: {error}")
                        logger.error(f"Tool output: {tool_output}")
                        logger.error(f"SQL query: {result.sql_query}")
                    return result.sql_query, tool_output, error

                sql_attempts = 0
                max_force_sql_attempts = 3
                error = True
                tool_output = None
                while sql_attempts < max_force_sql_attempts and error:
                    sql_attempts += 1

                    sql_query, tool_output, error = await query_llm_for_sql(feedback=tool_output)
                    if not error:
                        break

                if not error:
                    tool_output_handler = LoggingHandler(
                        prompt_file="sliders/answer_with_tool_use_output.prompt",
                        metadata={
                            "question": question,
                            "tool_call": sql_query,
                            "tool_output": json.dumps(tool_output),
                            "question_id": metadata.get("question_id", None),
                            "stage": "tool_output",
                        },
                    )
                    final_answer = await self.tool_output_chain.ainvoke(
                        {
                            "question": question,
                            "tool_call": sql_query,
                            "tool_output": json.dumps(tool_output),
                            "tables": tables_to_template_dicts(tables),
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [tool_output_handler]},
                    )

                else:
                    logger.info("Error running SQL query, generating direct answer...")
                    final_answer = "Error running SQL query"

        # Record answer generation timing
        metadata["timing"]["answer_generation"]["answer_time"] = time.time() - answer_start_time

        # Estimate final answer tokens (rough approximation)
        if isinstance(final_answer, str):
            metadata["answer_generation"]["final_answer_tokens"] = len(final_answer.split())

        return final_answer

    async def _generate_regular_answer(
        self, question: str, tables: list["Table"], schema: Tables, duck_sql_conn, metadata: dict
    ) -> str:
        """Generate answer using the regular approach (without inspect_answer)."""

        async def query_llm_for_sql(feedback: str = None):
            answer_question_handler = LoggingHandler(
                prompt_file="sliders/answer_from_schema_tool_use.prompt",
                metadata={
                    "question": question,
                    "classes": tables,
                    "feedback": feedback,
                    "question_id": metadata.get("question_id", None),
                    "stage": "answer_question_regular",
                },
            )
            result = await self.answer_question_chain.ainvoke(
                {
                    "tables": tables_to_template_dicts(tables),
                    "question": question,
                    "feedback": feedback,
                    "classes": prepare_schema_repr(schema),
                },
                config={"callbacks": [answer_question_handler]},
            )

            if result.run_sql:
                tool_output, error = run_sql_query(result.sql_query, duck_sql_conn)
                if error:
                    logger.error(f"Error running SQL query: {error}")
                    logger.error(f"SQL query: {result.sql_query}")
                return result, tool_output, error
            else:
                answer = result.answer
                if answer is None or (isinstance(answer, str) and answer.strip() == ""):
                    logger.warning("Answer chain returned no content while run_sql=False; requesting another attempt.")
                    feedback_message = (
                        "Previous attempt produced no answer. Use SQL if needed, otherwise provide a direct answer."
                    )
                    return result, feedback_message, True
                return result, None, False

        max_answer_question_attempts = 3
        error = True
        tool_output = None
        answer_question_attempts = 0

        def _is_status_400(exc: Exception) -> bool:
            code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
            if code == 400:
                return True
            msg = str(exc).lower()
            return "status code 400" in msg or "http 400" in msg

        while answer_question_attempts < max_answer_question_attempts and error:
            answer_question_attempts += 1
            logger.info(f"Regular answer attempts: {answer_question_attempts}")

            try:
                result, tool_output, error = await query_llm_for_sql(feedback=tool_output)
            except Exception as exc:
                if _is_status_400(exc):
                    logger.error(f"Answer generation failed with status 400: {exc}")
                    return "Input too large"
                raise
            if not error:
                break

        if error:
            direct_answer_handler = LoggingHandler(
                prompt_file="sliders/direct_answer_from_tables.prompt",
                metadata={
                    "question": question,
                    "classes": prepare_schema_repr(schema),
                    "question_id": metadata.get("question_id", None),
                    "stage": "direct_answer_regular",
                },
            )
            final_answer = await self.direct_answer_chain.ainvoke(
                {
                    "question": question,
                    "classes": prepare_schema_repr(schema),
                    "tables": tables_to_template_dicts(tables),
                },
                config={"callbacks": [direct_answer_handler]},
            )
        elif result.run_sql:
            tool_output_handler = LoggingHandler(
                prompt_file="sliders/answer_with_tool_use_output.prompt",
                metadata={
                    "question": question,
                    "tool_call": result.sql_query,
                    "tool_output": json.dumps(tool_output),
                    "question_id": metadata.get("question_id", None),
                    "stage": "tool_output_regular",
                },
            )
            final_answer = await self.tool_output_chain.ainvoke(
                {
                    "question": question,
                    "tool_call": result.sql_query,
                    "tool_output": json.dumps(tool_output),
                    "tables": tables_to_template_dicts(tables),
                    "classes": prepare_schema_repr(schema),
                },
                config={"callbacks": [tool_output_handler]},
            )
        else:
            final_answer = result.answer

        return final_answer

    async def _answer_question_from_tables(
        self, question: str, tables: list["Table"], schema: Tables, metadata: dict
    ) -> str:
        answer_start_time = time.time()

        with DuckSQLBasic() as duck_sql_conn:
            filtered_tables = []
            for table in tables:
                df = getattr(table, "dataframe", None)
                if df is None or getattr(df, "shape", (0, 0))[1] == 0:
                    logger.info(f"Skipping table {table.name} due to empty or missing dataframe.")
                    table.table_str = None
                    continue

                duck_sql_conn.register(
                    df,
                    table.dataframe_table_name,
                    schema=schema,
                    schema_table_name=table.name,
                )
                table.table_str = (
                    str(tuple(df.columns.to_list()))
                    + "\n"
                    + "\n".join([str(row) for row in df.to_records(index=False)])
                )
                filtered_tables.append(table)

            tables = filtered_tables

            # Check if SQL inspect answer strategy is enabled.
            # For pre-merge runs, keep only regular answer generation to avoid extra inspect pass.
            answer_stage = metadata.get("answer_stage", "")
            should_run_sql_inspect = self.config.get("answer_strategy") == "sql_inspect" and len(tables) > 0
            if answer_stage == "pre_merge":
                should_run_sql_inspect = False

            if should_run_sql_inspect:
                metadata["answer_generation"]["used_tables"] = True
                metadata["answer_generation"]["answer_strategy"] = "sql_inspect"

                # Extract reconciliation stats if available
                reconciliation_stats = {}
                if "reconciliation" in metadata and "stats" in metadata["reconciliation"]:
                    reconciliation_stats = metadata["reconciliation"]["stats"]

                # Extract inspect_answer config
                inspect_answer_config = self.config.get("inspect_answer", {})

                # First generate answer WITHOUT inspect_answer (regular approach)
                logger.info("=" * 80)
                logger.info("GENERATING ANSWER WITHOUT INSPECT_ANSWER (REGULAR APPROACH)")
                logger.info("=" * 80)

                regular_answer = await self._generate_regular_answer(
                    question=question,
                    tables=tables,
                    schema=schema,
                    duck_sql_conn=duck_sql_conn,
                    metadata=deepcopy(metadata),
                )

                logger.info("=" * 80)
                logger.info("REGULAR ANSWER (without inspect_answer):")
                logger.info(regular_answer)
                logger.info("=" * 80)

                # Then generate answer WITH inspect_answer
                logger.info("GENERATING ANSWER WITH INSPECT_ANSWER")
                logger.info("=" * 80)

                inspect_answer = await run_inspect_answer(
                    question=question,
                    tables=tables,
                    schema=schema,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                    model_config=self.config["models"],
                    tool_output_chain=self.tool_output_chain,
                    reconciliation_stats=reconciliation_stats,
                    inspect_answer_config=inspect_answer_config,
                )

                logger.info("=" * 80)
                logger.info("INSPECT ANSWER:")
                logger.info(inspect_answer)
                logger.info("=" * 80)

                # Store both answers in metadata
                metadata["regular_answer"] = regular_answer
                metadata["inspect_answer"] = inspect_answer

                return inspect_answer

            if self.config.get("no_sql", False):
                metadata["answer_generation"]["answer_strategy"] = "no_sql"
                if len(tables) == 0:
                    # No tables available – fall back to answering from schema/classes only
                    metadata["answer_generation"]["used_tables"] = False
                    answer_question_no_table_handler = LoggingHandler(
                        prompt_file="sliders/answer_from_schema_no_table.prompt",
                        metadata={
                            "question": question,
                            "classes": tables,
                            "feedback": None,
                            "question_id": metadata.get("question_id", None),
                            "stage": "answer_question_no_table",
                        },
                    )
                    result = await self.answer_question_no_table_chain.ainvoke(
                        {"question": question, "classes": tables, "feedback": None},
                        config={"callbacks": [answer_question_no_table_handler]},
                    )
                    final_answer = result.answer
                else:
                    # Tables are available – answer directly from tables without any SQL tool calls
                    metadata["answer_generation"]["used_tables"] = True
                    direct_answer_handler = LoggingHandler(
                        prompt_file="sliders/direct_answer_from_tables.prompt",
                        metadata={
                            "question": question,
                            "classes": prepare_schema_repr(schema),
                            "question_id": metadata.get("question_id", None),
                            "stage": "direct_answer_no_sql",
                        },
                    )
                    final_answer = await self.direct_answer_chain.ainvoke(
                        {
                            "question": question,
                            "classes": prepare_schema_repr(schema),
                            "tables": tables_to_template_dicts(tables),
                        },
                        config={"callbacks": [direct_answer_handler]},
                    )
            # If no tables, answer question from schema
            elif len(tables) == 0:
                metadata["answer_generation"]["used_tables"] = False
                answer_question_no_table_handler = LoggingHandler(
                    prompt_file="sliders/answer_from_schema_no_table.prompt",
                    metadata={
                        "question": question,
                        "classes": tables,
                        "feedback": None,
                        "question_id": metadata.get("question_id", None),
                        "stage": "answer_question_no_table",
                    },
                )
                result = await self.answer_question_no_table_chain.ainvoke(
                    {"question": question, "classes": tables, "feedback": None},
                    config={"callbacks": [answer_question_no_table_handler]},
                )
                final_answer = result.answer
            # If tables, answer question from tables
            else:

                async def query_llm_for_sql(feedback: str = None):
                    answer_question_handler = LoggingHandler(
                        prompt_file="sliders/answer_from_schema_tool_use.prompt",
                        metadata={
                            "question": question,
                            "classes": tables,
                            "feedback": feedback,
                            "question_id": metadata.get("question_id", None),
                            "stage": "answer_question",
                        },
                    )
                    result = await self.answer_question_chain.ainvoke(
                        {
                            "tables": tables_to_template_dicts(tables),
                            "question": question,
                            "feedback": feedback,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [answer_question_handler]},
                    )

                    if result.run_sql:
                        tool_output, error = run_sql_query(result.sql_query, duck_sql_conn)
                        if error:
                            logger.error(f"Error running SQL query: {error}")
                            logger.error(f"SQL query: {result.sql_query}")
                        return result, tool_output, error
                    else:
                        answer = result.answer
                        if answer is None or (isinstance(answer, str) and answer.strip() == ""):
                            logger.warning(
                                "Answer chain returned no content while run_sql=False; requesting another attempt."
                            )
                            feedback_message = (
                                "Previous attempt produced no answer. "
                                "Use SQL if needed, otherwise provide a direct answer."
                            )
                            return result, feedback_message, True
                        return result, None, False

                metadata["answer_generation"]["used_tables"] = True

                max_answer_question_attempts = 3
                error = True
                tool_output = None
                answer_question_attempts = 0
                while answer_question_attempts < max_answer_question_attempts and error:
                    answer_question_attempts += 1
                    logger.info(f"Answer question attempts: {answer_question_attempts}")

                    result, tool_output, error = await query_llm_for_sql(feedback=tool_output)
                    if not error:
                        break

                if error:
                    direct_answer_handler = LoggingHandler(
                        prompt_file="sliders/direct_answer_from_tables.prompt",
                        metadata={
                            "question": question,
                            "classes": prepare_schema_repr(schema),
                            "question_id": metadata.get("question_id", None),
                            "stage": "direct_answer",
                        },
                    )
                    final_answer = await self.direct_answer_chain.ainvoke(
                        {
                            "question": question,
                            "classes": prepare_schema_repr(schema),
                            "tables": tables_to_template_dicts(tables),
                        },
                        config={"callbacks": [direct_answer_handler]},
                    )
                elif result.run_sql:
                    tool_output_handler = LoggingHandler(
                        prompt_file="sliders/answer_with_tool_use_output.prompt",
                        metadata={
                            "question": question,
                            "tool_call": result.sql_query,
                            "tool_output": json.dumps(tool_output),
                            "question_id": metadata.get("question_id", None),
                            "stage": "tool_output",
                        },
                    )
                    final_answer = await self.tool_output_chain.ainvoke(
                        {
                            "question": question,
                            "tool_call": result.sql_query,
                            "tool_output": json.dumps(tool_output),
                            "tables": tables_to_template_dicts(tables),
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [tool_output_handler]},
                    )
                else:
                    final_answer = result.answer

        # Record answer generation timing
        metadata["timing"]["answer_generation"]["answer_time"] = time.time() - answer_start_time

        # Estimate final answer tokens (rough approximation)
        if isinstance(final_answer, str):
            metadata["answer_generation"]["final_answer_tokens"] = len(final_answer.split())

        return final_answer
