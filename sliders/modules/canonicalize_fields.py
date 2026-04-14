"""Field canonicalization module for automatically extracted tables."""

import asyncio
from typing import Literal, Optional
from pydantic import BaseModel

from sliders.llm_models import Tables
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.llm_tools.sql import run_sql_query
from sliders.llm_tools import DuckSQLBasic
from sliders.utils import get_table_stats, format_table_stats, prepare_schema_dict
import pandas as pd


class CanonicalizeDecision(BaseModel):
    """Decision from the canonicalizer: inspect, canonicalize, stop, or skip."""

    reasoning: str
    action: Literal["inspect", "canonicalize", "stop", "skip"]
    sql: Optional[str] = None  # SQL query for inspection or canonicalization


class FieldCanonicalizer:
    """Canonicalizes field values in extracted tables using LLM + SQL."""

    def __init__(self, model_config: dict, canonicalization_config: dict = None):
        """
        Initialize the field canonicalizer.

        Args:
            model_config: Model configuration dict with keys for different LLM tasks
            canonicalization_config: Configuration for canonicalization behavior (cycles, retries, etc.)
        """
        self.model_config = model_config
        self.canonicalization_config = canonicalization_config or {}

    async def canonicalize_table(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        duck_sql_conn,
        metadata: dict,
        inspections_per_field: int = 50,
        document_name: Optional[str] = None,
        max_cycles_override: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Canonicalize primary key fields in a table.

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names to canonicalize
            duck_sql_conn: DuckSQLBasic connection with registered table
            metadata: Metadata dictionary for logging
            inspections_per_field: Maximum inspections per field
            document_name: Optional document name for document-level canonicalization
            max_cycles_override: If provided, overrides the configured max_cycles limit

        Returns:
            Tuple of (canonicalized_dataframe, canonicalization_info)
        """
        df = table.dataframe
        table_name = table.dataframe_table_name

        if document_name:
            logger.info(f"=== Canonicalizing Primary Key Fields for document: {document_name} ===")
        else:
            logger.info(f"=== Canonicalizing Primary Key Fields in {table.name} ===")
        logger.info(f"Primary Key Fields: {primary_key}")
        logger.info(f"Fields to Process: {len(primary_key)}")

        # Initialize LLM chain - use document-level prompt if document_name provided
        model = self.model_config.get("canonicalize_fields", {}).get("model", "gpt-4.1")
        llm_client = get_llm_client(model=model)

        prompt_file = (
            "sliders/canonicalize_document_level.prompt" if document_name else "sliders/canonicalize_fields.prompt"
        )
        canonicalize_template = load_fewshot_prompt_template(
            template_file=prompt_file,
            template_blocks=[],
        )
        canonicalize_chain = canonicalize_template | llm_client.with_structured_output(CanonicalizeDecision)

        # Track canonicalization progress
        canonicalized_fields = []
        canonicalization_sqls = []
        skipped_fields = []
        current_df = df.copy()

        # Process each primary key field
        max_canonicalization_cycles = max_cycles_override or self.canonicalization_config.get("max_cycles", 20)
        max_inspections_per_cycle = inspections_per_field  # Reset per cycle
        max_retries_per_cycle = self.canonicalization_config.get("max_retries_per_cycle", 3)
        max_outer_iterations = self.canonicalization_config.get("max_outer_iterations", max_canonicalization_cycles * 2)
        max_inspection_cells = self.canonicalization_config.get("max_inspection_cells", 400)

        decision = None
        for field_idx, field_name in enumerate(primary_key):
            logger.info(f"\n--- Processing Field {field_idx + 1}/{len(primary_key)}: {field_name} ---")

            # State that persists across all canonicalization cycles for this field
            canonicalization_cycle = 0  # Number of canonicalization SQL attempts (successful or failed)
            outer_iterations = 0  # Total outer loop iterations (safety bound)
            successful_canonicalizations = []  # History of successful canonicalizations
            inspection_history = []  # Max entries, persists across cycles (unless cycle fails)
            max_inspection_history = self.canonicalization_config.get("max_inspection_history", 3)

            # Outer loop: up to max_canonicalization_cycles canonicalization cycles,
            # with max_outer_iterations as a hard safety bound on total iterations.
            while canonicalization_cycle < max_canonicalization_cycles:
                outer_iterations += 1
                if outer_iterations > max_outer_iterations:
                    logger.warning(
                        f"  Safety limit reached: {max_outer_iterations} outer loop iterations "
                        f"for field {field_name} without completing {max_canonicalization_cycles} "
                        f"canonicalization cycles (completed {canonicalization_cycle}). Stopping."
                    )
                    break

                logger.info(f"\n  Canonicalization Cycle {canonicalization_cycle + 1}/{max_canonicalization_cycles}")

                # State for current canonicalization cycle (resets each cycle)
                inspections_used = 0
                canonicalization_attempts = 0
                canonicalization_error_feedback = None

                # Inner loop: inspections + canonicalization attempts for this cycle
                while inspections_used <= max_inspections_per_cycle:
                    inspections_remaining = max_inspections_per_cycle - inspections_used

                    handler = LoggingHandler(
                        prompt_file="sliders/canonicalize_fields.prompt",
                        metadata={
                            "table_name": table_name,
                            "field_name": field_name,
                            "stage": f"field_{field_idx + 1}_cycle_{canonicalization_cycle}_inspection_{inspections_used}",
                            "question_id": metadata.get("question_id", None),
                            **(metadata or {}),
                        },
                    )

                    # Get current table stats
                    current_stats = get_table_stats(current_df, table_name)

                    # Format inspection history (max inspections_per_field entries)
                    formatted_inspection_history = "\n".join(
                        [
                            f"Inspection {i + 1}:\nSQL: {sql}\nResult:\n{result}\n"
                            for i, (sql, result) in enumerate(inspection_history)
                        ]
                    )

                    # Reset inspection history if the previous action was a successful canonicalization
                    if decision:
                        if (
                            canonicalization_cycle > 0
                            and decision.action == "canonicalize"
                            and successful_canonicalizations
                            and len(successful_canonicalizations) > canonicalization_cycle - 1
                        ):
                            # Clear inspection history for fresh start after successful canonicalization
                            inspection_history = []
                            formatted_inspection_history = "No inspections yet"

                    # Format canonicalization history (all successful ones)
                    # Don't show SQL details to avoid hallucination - just count
                    num_successful_canonicalizations = len(successful_canonicalizations)
                    formatted_canonicalization_history = (
                        f"{num_successful_canonicalizations} transformation(s) applied so far"
                    )

                    # Invoke LLM
                    invoke_params = {
                        "table_name": table_name,
                        "field_name": field_name,
                        "schema": self._format_schema(schema, table.name, current_df),
                        "table_stats": format_table_stats(current_stats),
                        "inspection_history": formatted_inspection_history
                        if formatted_inspection_history
                        else "No inspections yet",
                        "canonicalization_history": formatted_canonicalization_history,
                        "successful_canonicalization_count": num_successful_canonicalizations,
                        "canonicalization_cycles_remaining": max_canonicalization_cycles - canonicalization_cycle,
                        "inspections_remaining": inspections_remaining,
                        "inspections_per_field": max_inspections_per_cycle,
                        "max_inspection_cells": max_inspection_cells,
                        "canonicalization_error_feedback": canonicalization_error_feedback or "",
                    }
                    # Add document_name for document-level canonicalization
                    if document_name:
                        invoke_params["document_name"] = document_name

                    decision = await canonicalize_chain.ainvoke(
                        invoke_params,
                        config={"callbacks": [handler]},
                    )

                    logger.info(f"    Action: {decision.action}")
                    logger.info(f"    Reasoning: {decision.reasoning}")

                    if decision.action == "stop":
                        logger.info(f"  Agent chose to stop canonicalization for {field_name}")
                        logger.info(f"  Total successful canonicalizations: {len(successful_canonicalizations)}")
                        break  # Exit inner loop

                    elif decision.action == "skip":
                        logger.info(f"⊘ Skipping field: {field_name}")
                        skipped_fields.append(field_name)
                        break  # Exit inner loop

                    elif decision.action == "inspect":
                        if decision.sql and inspections_remaining > 0:
                            logger.info(f"    Running inspection {inspections_used + 1}/{max_inspections_per_cycle}")
                            logger.info(f"    SQL: {decision.sql}")
                            result_df, error = run_sql_query(decision.sql, duck_sql_conn, output_format="dataframe")

                            if not error:
                                if result_df is not None and not result_df.empty:
                                    num_columns = len(result_df.columns)
                                    max_rows = max(1, max_inspection_cells // num_columns)
                                    truncated = len(result_df) > max_rows
                                    display_df = result_df.head(max_rows) if truncated else result_df

                                    if truncated:
                                        result = (
                                            f"Showing first {max_rows} of {len(result_df)} rows "
                                            f"(cell limit: {max_inspection_cells} cells = "
                                            f"{max_rows} rows x {num_columns} columns). "
                                            f"Use more targeted queries or additional inspections to see remaining data.\n"
                                        )
                                    else:
                                        result = ""
                                    result += display_df.to_string(index=False)
                                else:
                                    result = "No results"

                                inspection_history.append((decision.sql, result))
                                if len(inspection_history) > max_inspection_history:
                                    inspection_history.pop(0)
                                logger.info(
                                    f"    ✓ Inspection successful (history size: {len(inspection_history)}/{max_inspection_history})"
                                )
                                inspections_used += 1
                            else:
                                logger.warning(f"    ✗ Inspection error: {result_df}")
                                inspections_used += 1  # Count failed attempts
                        else:
                            logger.warning("    No SQL provided or inspection budget exhausted")
                            break  # Exit inner loop

                    elif decision.action == "canonicalize":
                        if decision.sql:
                            canonicalization_attempts += 1
                            logger.info(
                                f"    Canonicalizing field: {field_name} (attempt {canonicalization_attempts}/{max_retries_per_cycle})"
                            )
                            logger.info(f"    SQL: {decision.sql}")

                            # Try to execute canonicalization
                            result_df, error = run_sql_query(decision.sql, duck_sql_conn, output_format="dataframe")

                            if not error and result_df is not None:
                                # Success!
                                current_df = result_df
                                # Re-register updated table
                                duck_sql_conn.register(
                                    current_df,
                                    table_name,
                                    schema=schema,
                                    schema_table_name=table.name,
                                )

                                # Store successful canonicalization
                                successful_canonicalizations.append(
                                    {
                                        "cycle": canonicalization_cycle + 1,
                                        "sql": decision.sql,
                                        "reasoning": decision.reasoning,
                                    }
                                )

                                logger.info(
                                    f"    ✓ Successfully applied canonicalization {len(successful_canonicalizations)}"
                                )

                                # Clear error feedback on success
                                canonicalization_error_feedback = None

                                # Move to next cycle
                                canonicalization_cycle += 1
                                break  # Exit inner loop, start next cycle
                            else:
                                # Canonicalization failed
                                logger.warning(
                                    f"    ✗ Canonicalization attempt {canonicalization_attempts}/{max_retries_per_cycle} failed"
                                )
                                logger.warning(f"       Error: {result_df}")

                                # Check if we've exhausted retries for this cycle
                                if canonicalization_attempts >= max_retries_per_cycle:
                                    logger.warning(
                                        f"    ✗ Exhausted {max_retries_per_cycle} retries for this canonicalization cycle"
                                    )

                                    # Increment cycle counter (failed cycle counts toward the 10)
                                    canonicalization_cycle += 1

                                    # Erase inspection history on failed cycle
                                    logger.info("    Erasing inspection history due to failed cycle")
                                    inspection_history = []

                                    # Clear error feedback
                                    canonicalization_error_feedback = None

                                    break  # Exit inner loop, start next cycle

                                # Build error feedback for next iteration
                                canonicalization_error_feedback = (
                                    f"SQL: {decision.sql}\n\n"
                                    f"Error: {result_df}\n\n"
                                    f"Attempt {canonicalization_attempts}/{max_retries_per_cycle} in cycle {canonicalization_cycle + 1}. Please fix the SQL syntax and try again."
                                )

                                # Continue inner loop to retry
                                continue
                        else:
                            logger.warning("    No SQL provided for canonicalization")
                            break  # Exit inner loop

                # Check if we should exit outer loop (stop or skip action)
                if decision.action in ["stop", "skip"]:
                    break  # Exit outer loop

            # After all cycles complete for this field
            if successful_canonicalizations:
                canonicalized_fields.append(field_name)
                # Store all successful canonicalizations for this field
                for canonicalization_info in successful_canonicalizations:
                    canonicalization_sqls.append(
                        {
                            "field": field_name,
                            "cycle": canonicalization_info["cycle"],
                            "sql": canonicalization_info["sql"],
                            "reasoning": canonicalization_info["reasoning"],
                        }
                    )
                logger.info(
                    f"\n  ✓ Field {field_name} canonicalized with {len(successful_canonicalizations)} transformation(s)"
                )
            elif field_name not in skipped_fields:
                logger.warning(
                    f"\n  ⊘ Field {field_name} not canonicalized - {canonicalization_cycle} cycles completed without success"
                )

        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("CANONICALIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Primary Key Fields Processed: {len(primary_key)}")
        logger.info(f"Fields Canonicalized: {len(canonicalized_fields)}")
        if canonicalized_fields:
            for field_info in canonicalization_sqls:
                logger.info(f"  ✓ {field_info['field']}: {field_info['reasoning'][:80]}...")
        logger.info(f"Fields Skipped: {len(skipped_fields)}")
        if skipped_fields:
            for field in skipped_fields:
                logger.info(f"  ⊘ {field}")
        logger.info("=" * 80)

        canonicalization_info = {
            "canonicalized_fields": canonicalized_fields,
            "skipped_fields": skipped_fields,
            "canonicalization_count": len(canonicalized_fields),
            "canonicalization_sqls": canonicalization_sqls,
        }

        return current_df, canonicalization_info

    async def _canonicalize_single_document(
        self,
        doc_name: str,
        doc_df: pd.DataFrame,
        table_name: str,
        original_table_name: str,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        inspections_per_field: int,
    ) -> tuple[str, pd.DataFrame, dict]:
        """
        Canonicalize a single document's data. Uses its own DuckDB connection for thread safety.

        Args:
            doc_name: Name of the document
            doc_df: DataFrame containing only this document's rows
            table_name: Base table name
            original_table_name: Original schema table name
            schema: Schema information
            primary_key: List of primary key field names
            metadata: Metadata dictionary for logging
            inspections_per_field: Maximum inspections per field

        Returns:
            Tuple of (doc_name, canonicalized_dataframe, canonicalization_info)
        """
        from sliders.llm_models import ExtractedTable

        doc_table_name = f"{table_name}_doc_{hash(str(doc_name)) % 10000}"

        # Create temporary ExtractedTable for this document
        doc_table = ExtractedTable(
            name=original_table_name,
            tables=schema,
            sql_query=None,
            dataframe=doc_df,
            dataframe_table_name=doc_table_name,
            table_str=str(doc_df),
        )

        # Each document gets its own DuckDB connection for parallel safety
        with DuckSQLBasic() as doc_duck_sql_conn:
            # Register document-specific table
            doc_duck_sql_conn.register(doc_df, doc_table_name, schema=schema, schema_table_name=original_table_name)

            try:
                doc_max_cycles = self.canonicalization_config.get("document_level_max_cycles", 5)
                canonicalized_df, doc_canon_info = await self.canonicalize_table(
                    doc_table,
                    schema,
                    primary_key,
                    doc_duck_sql_conn,
                    metadata,
                    inspections_per_field,
                    document_name=str(doc_name),
                    max_cycles_override=doc_max_cycles,
                )
                return (str(doc_name), canonicalized_df, doc_canon_info)
            except Exception as e:
                logger.error(f"Error canonicalizing document {doc_name}: {e}")
                # Return original data on error
                return (str(doc_name), doc_df, {"error": str(e)})

    async def canonicalize_table_two_pass(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        duck_sql_conn,
        metadata: dict,
        inspections_per_field: int = 50,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Two-pass canonicalization:
        1. Document-level: Canonicalize within each document independently
        2. Global-level: Canonicalize the merged result to catch cross-document variations

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names to canonicalize
            duck_sql_conn: DuckSQLBasic connection with registered table
            metadata: Metadata dictionary for logging
            inspections_per_field: Maximum inspections per field

        Returns:
            Tuple of (canonicalized_dataframe, two_pass_info)
        """
        from sliders.llm_models import ExtractedTable

        logger.info("=" * 80)
        logger.info("TWO-PASS CANONICALIZATION START")
        logger.info("=" * 80)

        two_pass_info = {
            "pass_1_document_level": {},
            "pass_2_global_level": {},
        }

        # === PASS 1: Document-level canonicalization ===
        logger.info("\n=== PASS 1: DOCUMENT-LEVEL CANONICALIZATION ===")
        doc_canonicalized_df, doc_level_info = await self.canonicalize_table_by_document(
            table, schema, primary_key, metadata, inspections_per_field
        )
        two_pass_info["pass_1_document_level"] = doc_level_info

        # === PASS 2: Global canonicalization ===
        logger.info("\n=== PASS 2: GLOBAL CANONICALIZATION ===")

        # Create new ExtractedTable with document-canonicalized data
        global_table = ExtractedTable(
            name=table.name,
            tables=schema,
            sql_query=None,
            dataframe=doc_canonicalized_df,
            dataframe_table_name=table.dataframe_table_name,
            table_str=str(doc_canonicalized_df),
        )

        # Re-register the table with document-canonicalized data
        duck_sql_conn.register(
            doc_canonicalized_df,
            table.dataframe_table_name,
            schema=schema,
            schema_table_name=table.name,
        )

        final_df, global_level_info = await self.canonicalize_table(
            global_table, schema, primary_key, duck_sql_conn, metadata, inspections_per_field
        )
        two_pass_info["pass_2_global_level"] = global_level_info

        logger.info("=" * 80)
        logger.info("TWO-PASS CANONICALIZATION COMPLETE")
        logger.info("=" * 80)

        return final_df, two_pass_info

    def _format_schema(self, schema: Tables, table_name: str, df: pd.DataFrame = None) -> str:
        """Format the schema for a specific table.

        Args:
            schema: The Tables schema object
            table_name: Name of the table
            df: Optional dataframe to filter schema to only include columns that exist in the dataframe
        """
        schema_list = prepare_schema_dict(schema)

        # Get actual columns from dataframe if provided
        actual_columns = set(df.columns) if df is not None else None

        for table_schema in schema_list:
            if table_schema.get("name") == table_name:
                fields_desc = []
                for field in table_schema.get("fields", []):
                    field_name = field["name"]
                    # Skip fields that don't exist in the actual dataframe
                    if actual_columns is not None and field_name not in actual_columns:
                        continue
                    field_info = f"  - {field_name}"
                    if field.get("description"):
                        field_info += f": {field['description']}"
                    fields_desc.append(field_info)

                # Also add columns that exist in dataframe but not in schema
                if actual_columns is not None:
                    schema_field_names = {f["name"] for f in table_schema.get("fields", [])}
                    for col in actual_columns:
                        if col not in schema_field_names and not col.startswith("__"):
                            fields_desc.append(f"  - {col}")

                return f"Table: {table_name}\nFields:\n" + "\n".join(fields_desc)

        # Fallback: if schema not found but dataframe provided, list dataframe columns
        if df is not None:
            fields_desc = [f"  - {col}" for col in df.columns if not col.startswith("__")]
            return f"Table: {table_name}\nFields:\n" + "\n".join(fields_desc)

        return f"Table: {table_name}\n(Schema not found)"

    async def _canonicalize_single_document(
        self,
        document_name: str,
        document_df: pd.DataFrame,
        table,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        inspections_per_field: int,
    ) -> tuple[str, pd.DataFrame]:
        """
        Canonicalize a single document's data.

        Args:
            document_name: Name of the document
            document_df: DataFrame containing only this document's rows
            table: Original table object (for schema info)
            schema: Schema information
            primary_key: List of primary key field names to canonicalize
            metadata: Metadata dictionary for logging
            inspections_per_field: Maximum inspections per field

        Returns:
            Tuple of (document_name, canonicalized_dataframe)
        """
        from sliders.llm_models import ExtractedTable
        from sliders.llm_tools.sql import DuckSQLBasic
        import re

        # Sanitize document name for SQL table name (replace special chars with underscores)
        # Keep only alphanumeric and underscores
        sanitized_doc_name = re.sub(r"[^a-zA-Z0-9_]", "_", document_name)
        # Ensure it doesn't start with a number
        if sanitized_doc_name and sanitized_doc_name[0].isdigit():
            sanitized_doc_name = f"doc_{sanitized_doc_name}"

        doc_table_name = f"{table.dataframe_table_name}_{sanitized_doc_name}"

        logger.debug(
            f"Processing document '{document_name}' -> sanitized table name: '{doc_table_name}' "
            f"({len(document_df)} rows)"
        )

        # Create a temporary table object for this document
        doc_table = ExtractedTable(
            name=table.name,
            tables=schema,
            sql_query=None,
            dataframe=document_df,
            dataframe_table_name=doc_table_name,
            table_str=str(document_df),
        )

        # Create a new DuckSQL connection for this document
        duck_sql_conn = DuckSQLBasic()
        duck_sql_conn.register(
            document_df,
            doc_table.dataframe_table_name,
            schema=schema,
            schema_table_name=table.name,
        )

        doc_metadata = {**metadata, "document_name": document_name}
        doc_max_cycles = self.canonicalization_config.get("document_level_max_cycles", 5)

        canonicalized_df, _ = await self.canonicalize_table(
            table=doc_table,
            schema=schema,
            primary_key=primary_key,
            duck_sql_conn=duck_sql_conn,
            metadata=doc_metadata,
            inspections_per_field=inspections_per_field,
            document_name=document_name,
            max_cycles_override=doc_max_cycles,
        )

        return document_name, canonicalized_df

    async def canonicalize_table_by_document(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        inspections_per_field: int = 50,
        document_column: str = "document_name",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Canonicalize a table by processing each document in parallel.

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names to canonicalize
            metadata: Metadata dictionary for logging
            inspections_per_field: Maximum inspections per field
            document_column: Column name containing document identifiers

        Returns:
            Tuple of (canonicalized_dataframe, canonicalization_info)
        """
        df = table.dataframe

        if document_column not in df.columns:
            logger.warning(
                f"Document column '{document_column}' not found. Falling back to single-table canonicalization."
            )
            from sliders.llm_tools.sql import DuckSQLBasic

            duck_sql_conn = DuckSQLBasic()
            duck_sql_conn.register(
                df,
                table.dataframe_table_name,
                schema=schema,
                schema_table_name=table.name,
            )
            return await self.canonicalize_table(
                table=table,
                schema=schema,
                primary_key=primary_key,
                duck_sql_conn=duck_sql_conn,
                metadata=metadata,
                inspections_per_field=inspections_per_field,
            )

        # Group by document
        documents = df[document_column].unique()
        logger.info(f"Found {len(documents)} documents to canonicalize in parallel")

        # Create tasks for each document, tracking original data for fallback
        tasks = []
        document_original_dfs = {}  # Map document_name -> original DataFrame

        for document_name in documents:
            document_df = df[df[document_column] == document_name].copy()
            document_original_dfs[document_name] = document_df

            task = self._canonicalize_single_document(
                document_name=document_name,
                document_df=document_df,
                table=table,
                schema=schema,
                primary_key=primary_key,
                metadata=metadata,
                inspections_per_field=inspections_per_field,
            )
            tasks.append((document_name, task))

        # Run all document canonicalizations in parallel
        logger.info(f"Starting parallel canonicalization of {len(tasks)} documents...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Collect results, preserving original rows on exception
        all_canonicalized_dfs = []
        failed_documents = []
        successful_documents = []

        for (document_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Document canonicalization failed for '{document_name}': {type(result).__name__}: {result}"
                )
                logger.error(
                    f"  Preserving original {len(document_original_dfs[document_name])} rows for this document"
                )
                if hasattr(result, "__traceback__"):
                    import traceback

                    logger.debug(f"  Traceback: {''.join(traceback.format_tb(result.__traceback__))}")
                # Preserve original rows on failure
                all_canonicalized_dfs.append(document_original_dfs[document_name])
                failed_documents.append(document_name)
            else:
                _, canonicalized_df = result
                all_canonicalized_dfs.append(canonicalized_df)
                successful_documents.append(document_name)

        # Concatenate all results
        if all_canonicalized_dfs:
            final_df = pd.concat(all_canonicalized_dfs, ignore_index=True)
        else:
            final_df = df.copy()

        logger.info(
            f"Parallel canonicalization complete: {len(successful_documents)} succeeded, "
            f"{len(failed_documents)} failed (original rows preserved)"
        )

        canonicalization_info = {
            "successful_documents": successful_documents,
            "failed_documents": failed_documents,
            "total_documents": len(documents),
            "rows_processed": len(final_df),
        }

        return final_df, canonicalization_info
