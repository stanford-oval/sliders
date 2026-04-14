"""NULL Primary Key Handler module for automatically extracted tables.

This module handles rows with NULL values in primary key columns by:
1. Iterating through NULL patterns (all NULLs -> 1 NULL)
2. Partitioning rows by non-NULL PK values
3. Adding context rows for reference
4. Using LLM agent to fill, extract, discard, or add placeholders
"""

from typing import Literal, Optional
from pydantic import BaseModel
from itertools import combinations
import asyncio
import pandas as pd
import duckdb

from sliders.llm_models import Tables
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.utils import (
    get_table_stats,
    format_table_stats,
    format_table_schema,
    format_sql_result,
    format_dataframe_schema,
)


class NullPKDecision(BaseModel):
    """Decision from the NULL PK handler: inspect, process, approve, or regenerate."""

    reasoning: str
    action: Literal["inspect", "process", "approve", "regenerate"]
    sql: Optional[str] = None  # SQL query for inspection, processing, or regeneration
    row_decisions: Optional[list[dict]] = None  # List of decisions per NULL row


class NullPKHandler:
    """Handles NULL values in primary key columns using LLM + SQL."""

    def __init__(self, model_config: dict, null_pk_config: dict = None):
        """
        Initialize the NULL PK handler.

        Args:
            model_config: Model configuration dict with keys for different LLM tasks
            null_pk_config: Configuration for NULL handling behavior
        """
        self.model_config = model_config
        self.null_pk_config = null_pk_config or {}

    async def handle_null_primary_keys(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        duck_sql_conn,
        metadata: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Handle NULL values in primary key columns.

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names
            duck_sql_conn: DuckSQLBasic connection with registered table
            metadata: Metadata dictionary for logging

        Returns:
            Tuple of (processed_dataframe, null_handling_info)
        """
        df = table.dataframe
        table_name = table.dataframe_table_name

        logger.info(f"=== Handling NULL Primary Keys in {table.name} ===")
        logger.info(f"Primary Key: {primary_key}")
        logger.info(f"Initial rows: {len(df)}")

        # Track statistics
        total_null_rows = 0
        filled_count = 0
        discarded_count = 0
        placeholder_count = 0

        # Working dataframe
        current_df = df.copy()

        # Iterate from all NULLs down to 1 NULL
        for null_count in range(len(primary_key), 0, -1):
            logger.info(f"\n--- Processing NULL patterns with {null_count} NULL column(s) ---")

            # Get all combinations of null_count columns
            for null_cols_combo in combinations(primary_key, null_count):
                null_cols = list(null_cols_combo)
                non_null_cols = [c for c in primary_key if c not in null_cols]

                logger.info(f"\nPattern: NULL columns = {null_cols}, non-NULL columns = {non_null_cols}")

                # Process this NULL pattern
                current_df, pattern_stats = await self._process_null_pattern(
                    current_df=current_df,
                    table_name=table_name,
                    schema=schema,
                    primary_key=primary_key,
                    null_cols=null_cols,
                    non_null_cols=non_null_cols,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                )

                # Update statistics
                total_null_rows += pattern_stats.get("null_rows_processed", 0)
                filled_count += pattern_stats.get("filled_count", 0)
                discarded_count += pattern_stats.get("discarded_count", 0)
                placeholder_count += pattern_stats.get("placeholder_count", 0)

        logger.info("\n=== NULL PK Handling Complete ===")
        logger.info(f"Total NULL rows processed: {total_null_rows}")
        logger.info(f"  Filled: {filled_count}")
        logger.info(f"  Discarded: {discarded_count}")
        logger.info(f"  Placeholders: {placeholder_count}")
        logger.info(f"Final rows: {len(current_df)}")

        null_handling_info = {
            "total_null_rows": total_null_rows,
            "filled_count": filled_count,
            "discarded_count": discarded_count,
            "placeholder_count": placeholder_count,
            "initial_rows": len(df),
            "final_rows": len(current_df),
        }

        return current_df, null_handling_info

    async def _process_null_pattern(
        self,
        current_df: pd.DataFrame,
        table_name: str,
        schema: Tables,
        primary_key: list[str],
        null_cols: list[str],
        non_null_cols: list[str],
        duck_sql_conn,
        metadata: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Process a single NULL pattern across all partitions in parallel."""

        # Build filter mask for this NULL pattern using pandas boolean indexing
        mask = pd.Series([True] * len(current_df), index=current_df.index)

        # Apply NULL conditions
        for col in null_cols:
            mask &= current_df[col].isna()

        # Apply non-NULL conditions
        for col in non_null_cols:
            mask &= current_df[col].notna()

        # Filter to rows with this NULL pattern
        null_rows = current_df[mask]

        if len(null_rows) == 0:
            logger.info("  No rows found with this NULL pattern")
            return current_df, {
                "null_rows_processed": 0,
                "filled_count": 0,
                "discarded_count": 0,
                "placeholder_count": 0,
            }

        logger.info(f"  Found {len(null_rows)} rows with this NULL pattern")

        # Partition by non-NULL PK values
        if non_null_cols:
            partitions = {}
            for partition_value, null_rows_partition in null_rows.groupby(non_null_cols, dropna=False):
                # Ensure partition_value is tuple
                if not isinstance(partition_value, tuple):
                    partition_value = (partition_value,)

                # Get context rows: same non-NULL PK values but all PK columns non-NULL
                # Build mask for all PK columns being non-NULL
                context_mask = pd.Series([True] * len(current_df), index=current_df.index)
                for col in primary_key:
                    context_mask &= current_df[col].notna()

                context_rows = current_df[context_mask]

                # Filter context rows to match non-NULL PK values
                for i, col in enumerate(non_null_cols):
                    context_rows = context_rows[context_rows[col] == partition_value[i]]

                partitions[partition_value] = (null_rows_partition, context_rows)
        else:
            # All PK columns are NULL - single partition with no context
            partitions = {tuple(): (null_rows, pd.DataFrame(columns=current_df.columns))}

        logger.info(f"  Created {len(partitions)} partition(s) for parallel processing")

        if len(partitions) == 0:
            return current_df, {
                "null_rows_processed": 0,
                "filled_count": 0,
                "discarded_count": 0,
                "placeholder_count": 0,
            }

        # Process all partitions in PARALLEL
        tasks = [
            self._process_single_partition(
                null_rows=null_rows_partition,
                context_rows=context_rows,
                null_cols=null_cols,
                non_null_cols=non_null_cols,
                partition_value=partition_value,
                table_name=table_name,
                schema=schema,
                primary_key=primary_key,
                metadata=metadata,
            )
            for partition_value, (null_rows_partition, context_rows) in partitions.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results and print buffered logs sequentially
        all_processed_rows = []
        total_filled = 0
        total_discarded = 0
        total_placeholder = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Partition processing failed: {result}")
                # On failure, keep original NULL rows
                continue
            else:
                partition_value, processed_null_rows, context_rows, success, stats, log_buffer = result

                # Print buffered logs for this partition
                for log_msg in log_buffer:
                    logger.info(log_msg)

                if success:
                    all_processed_rows.append(processed_null_rows)
                    # Note: context_rows are already in current_df, no need to add them back
                    total_filled += stats.get("filled", 0)
                    total_discarded += stats.get("discarded", 0)
                    total_placeholder += stats.get("placeholder", 0)
                else:
                    logger.warning(f"Partition {partition_value} processing failed, keeping original rows")

        # Remove original NULL rows from current_df
        # (they will be replaced by processed versions)
        # Context rows remain as-is in the table
        remaining_df = current_df[~current_df.index.isin(null_rows.index)]

        # Add back only the processed NULL rows (context rows already exist in remaining_df)
        if all_processed_rows:
            processed_df = pd.concat([remaining_df] + all_processed_rows, ignore_index=True)
        else:
            processed_df = remaining_df

        pattern_stats = {
            "null_rows_processed": len(null_rows),
            "filled_count": total_filled,
            "discarded_count": total_discarded,
            "placeholder_count": total_placeholder,
        }

        return processed_df, pattern_stats

    async def _process_single_partition(
        self,
        null_rows: pd.DataFrame,
        context_rows: pd.DataFrame,
        null_cols: list[str],
        non_null_cols: list[str],
        partition_value: tuple,
        table_name: str,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        mode: str = "global",  # "global" or "document_level"
    ) -> tuple[tuple, pd.DataFrame, pd.DataFrame, bool, dict, list[str]]:
        """Process a single partition with LLM agent and optional verification.

        Args:
            mode: Processing mode - "global" for pattern-based, "document_level" for document-based

        Returns:
            tuple: (partition_value, processed_null_rows, context_rows, success, stats, log_messages)
        """

        # Buffer log messages for this partition
        log_buffer = []

        # Create a partition identifier for logging
        partition_id = f"[Partition {partition_value}]"

        log_buffer.append(f"\n{'=' * 80}")
        log_buffer.append(
            f"{partition_id} Processing: {len(null_rows)} NULL row(s), {len(context_rows)} context row(s)"
        )
        log_buffer.append(f"{'=' * 80}")

        # Get verification config
        verification_config = self.null_pk_config.get("verification", {})
        enable_verification = verification_config.get("enable", False)

        try:
            # Initialize LLM chain
            model = self.model_config.get("handle_null_pks", {}).get("model", "gpt-4.1")
            llm_client = get_llm_client(model=model, temperature=0.0)

            null_pk_template = load_fewshot_prompt_template(
                template_file="sliders/handle_null_primary_keys.prompt",
                template_blocks=[],
            )
            null_pk_chain = null_pk_template | llm_client.with_structured_output(
                NullPKDecision, method="function_calling"
            )

            # Create temporary DuckDB connection for this partition
            conn = duckdb.connect()
            try:
                # Register both tables
                conn.register("null_rows", null_rows)
                conn.register("context_rows", context_rows)

                # Inspection loop
                max_inspections = self.null_pk_config.get("max_inspections", 5)
                inspection_history = []

                for inspection_num in range(max_inspections + 1):
                    inspections_remaining = max_inspections - inspection_num
                    must_process_now = inspection_num == max_inspections

                    handler = LoggingHandler(
                        prompt_file="sliders/handle_null_primary_keys.prompt",
                        metadata={
                            "table_name": table_name,
                            "partition_value": str(partition_value),
                            "stage": f"null_pk_partition_{partition_value}_inspection_{inspection_num}",
                            "question_id": metadata.get("question_id", None),
                            **(metadata or {}),
                        },
                    )

                    # Format inspection history
                    formatted_history = "\n".join(
                        [
                            f"Inspection {i + 1}:\nSQL: {sql}\nResult:\n{result}\n"
                            for i, (sql, result) in enumerate(inspection_history)
                        ]
                    )

                    # Get table stats
                    null_stats = get_table_stats(null_rows, "null_rows")
                    context_stats = get_table_stats(context_rows, "context_rows")

                    decision = await null_pk_chain.ainvoke(
                        {
                            "table_name": table_name,
                            "schema": format_table_schema(schema, table_name),
                            "primary_key": primary_key,
                            "null_cols": null_cols,
                            "non_null_cols": non_null_cols,
                            "partition_value": partition_value,
                            "null_rows_count": len(null_rows),
                            "context_rows_count": len(context_rows),
                            "null_rows_stats": format_table_stats(null_stats),
                            "context_rows_stats": format_table_stats(context_stats),
                            "inspection_history": formatted_history if formatted_history else "No inspections yet",
                            "inspections_remaining": inspections_remaining,
                            "must_process_now": must_process_now,
                            "placeholder_text": self.null_pk_config.get("placeholder_text", "UNKNOWN"),
                            "mode": mode,  # Pass mode to prompt
                            # Verification mode variables
                            "verification_mode": False,
                            "initial_schema": "",
                            "final_schema": "",
                            "generated_sql": "",
                            "generation_reasoning": "",
                            "initial_row_count": 0,
                            "final_row_count": 0,
                            "remaining_inspections": 0,
                            "verification_inspection_history": "",
                        },
                        config={"callbacks": [handler]},
                    )

                    if decision.action == "inspect":
                        if decision.sql and inspections_remaining > 0:
                            log_buffer.append(f"{partition_id} Inspection {inspection_num + 1}/{max_inspections}")
                            log_buffer.append(f"{partition_id} SQL: {decision.sql}")
                            try:
                                result = conn.execute(decision.sql).fetchdf()
                                formatted_result = format_sql_result(result)
                                inspection_history.append((decision.sql, formatted_result))
                                log_buffer.append(f"{partition_id} ✓ Inspection successful")
                            except Exception as e:
                                error_msg = f"SQL Error: {str(e)}"
                                log_buffer.append(f"{partition_id} ✗ Inspection failed: {e}")
                                inspection_history.append((decision.sql, error_msg))
                        else:
                            log_buffer.append(f"{partition_id} No SQL provided or inspection budget exhausted")
                            break

                    elif decision.action == "process":
                        log_buffer.append(f"{partition_id} Agent ready to process NULL rows")
                        log_buffer.append(f"{partition_id} Reasoning: {decision.reasoning}")

                        # Execute processing SQL
                        if decision.sql:
                            log_buffer.append(f"{partition_id} Processing SQL: {decision.sql}")
                            try:
                                processed_null_rows = conn.execute(decision.sql).fetchdf()
                                log_buffer.append(
                                    f"{partition_id} ✓ Processing successful: {len(null_rows)} → {len(processed_null_rows)} rows"
                                )

                                # If verification is disabled, return immediately
                                if not enable_verification:
                                    # Count filled, discarded, placeholder
                                    filled = sum(
                                        1
                                        for _, row in processed_null_rows.iterrows()
                                        if all(pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols)
                                    )
                                    placeholder = sum(
                                        1
                                        for _, row in processed_null_rows.iterrows()
                                        if any(row[col] == "UNKNOWN" for col in null_cols)
                                    )
                                    discarded = len(null_rows) - len(processed_null_rows)

                                    stats = {"filled": filled, "discarded": discarded, "placeholder": placeholder}

                                    log_buffer.append(
                                        f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                    )
                                    log_buffer.append(f"{'=' * 80}\n")
                                    return (partition_value, processed_null_rows, context_rows, True, stats, log_buffer)

                                # VERIFICATION PHASE
                                log_buffer.append(f"{partition_id} Verification phase: Entering verification")
                                initial_table = null_rows.copy()
                                final_table = processed_null_rows
                                generated_sql = decision.sql
                                generation_reasoning = decision.reasoning

                                max_verif_inspections = verification_config.get("max_inspections", 5)
                                verification_inspection_count = 0
                                verification_inspection_history = []

                                # Register verification tables
                                conn.register("initial_table", initial_table)
                                conn.register("final_table", final_table)

                                while verification_inspection_count < max_verif_inspections:
                                    # Format verification inspection history
                                    formatted_verif_inspections = []
                                    for idx, insp in enumerate(verification_inspection_history):
                                        formatted_verif_inspections.append(
                                            f"Inspection {idx + 1}:\nReasoning: {insp['reasoning']}\nSQL: {insp['query']}\nResult:\n{insp['result']}\n"
                                        )

                                    handler = LoggingHandler(
                                        prompt_file="sliders/handle_null_primary_keys.prompt",
                                        metadata={
                                            "table_name": table_name,
                                            "partition_value": str(partition_value),
                                            "stage": f"null_pk_partition_{partition_value}_verification_{verification_inspection_count}",
                                            "question_id": metadata.get("question_id", None),
                                            **(metadata or {}),
                                        },
                                    )

                                    verification_decision = await null_pk_chain.ainvoke(
                                        {
                                            "partition_value": partition_value,
                                            "initial_schema": format_dataframe_schema(initial_table),
                                            "final_schema": format_dataframe_schema(final_table),
                                            "verification_mode": True,
                                            "generated_sql": generated_sql,
                                            "generation_reasoning": generation_reasoning,
                                            "initial_row_count": len(initial_table),
                                            "final_row_count": len(final_table),
                                            "remaining_inspections": max_verif_inspections
                                            - verification_inspection_count,
                                            "verification_inspection_history": "\n".join(formatted_verif_inspections)
                                            if formatted_verif_inspections
                                            else "No verification inspections yet",
                                            # Regular mode variables (required by template)
                                            "table_name": table_name,
                                            "schema": format_table_schema(schema, table_name),
                                            "primary_key": primary_key,
                                            "null_cols": null_cols,
                                            "non_null_cols": non_null_cols,
                                            "null_rows_count": 0,
                                            "context_rows_count": 0,
                                            "null_rows_stats": "",
                                            "context_rows_stats": "",
                                            "inspection_history": "",
                                            "inspections_remaining": 0,
                                            "must_process_now": False,
                                            "placeholder_text": "",
                                        },
                                        config={"callbacks": [handler]},
                                    )

                                    if verification_decision.action == "inspect":
                                        # Run inspection query on both tables
                                        try:
                                            log_buffer.append(
                                                f"{partition_id} Verification: Inspection {verification_inspection_count + 1}/{max_verif_inspections}"
                                            )
                                            log_buffer.append(
                                                f"{partition_id} Verification: Reasoning: {verification_decision.reasoning}"
                                            )
                                            log_buffer.append(
                                                f"{partition_id} Verification: SQL: {verification_decision.sql}"
                                            )

                                            result = conn.execute(verification_decision.sql).fetchdf()
                                            formatted_result = format_sql_result(result)
                                            verification_inspection_history.append(
                                                {
                                                    "query": verification_decision.sql,
                                                    "result": formatted_result,
                                                    "reasoning": verification_decision.reasoning,
                                                }
                                            )
                                            verification_inspection_count += 1
                                        except Exception as e:
                                            error_msg = f"SQL Error: {str(e)}"
                                            log_buffer.append(f"{partition_id} Verification: ✗ Inspection failed: {e}")
                                            verification_inspection_history.append(
                                                {
                                                    "query": verification_decision.sql,
                                                    "result": error_msg,
                                                    "reasoning": verification_decision.reasoning,
                                                }
                                            )
                                            verification_inspection_count += 1

                                    elif verification_decision.action == "approve":
                                        # Agent approved the final_table
                                        log_buffer.append(
                                            f"{partition_id} Verification: ✓ Approved after {verification_inspection_count} inspection(s)"
                                        )
                                        log_buffer.append(
                                            f"{partition_id} Verification: {verification_decision.reasoning}"
                                        )

                                        # Count filled, discarded, placeholder
                                        filled = sum(
                                            1
                                            for _, row in final_table.iterrows()
                                            if all(pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols)
                                        )
                                        placeholder = sum(
                                            1
                                            for _, row in final_table.iterrows()
                                            if any(row[col] == "UNKNOWN" for col in null_cols)
                                        )
                                        discarded = len(null_rows) - len(final_table)

                                        stats = {"filled": filled, "discarded": discarded, "placeholder": placeholder}

                                        log_buffer.append(
                                            f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                        )
                                        log_buffer.append(f"{'=' * 80}\n")
                                        return (partition_value, final_table, context_rows, True, stats, log_buffer)

                                    elif verification_decision.action == "regenerate":
                                        # Agent chose to regenerate
                                        log_buffer.append(
                                            f"{partition_id} Verification: Regenerating after {verification_inspection_count} inspection(s)"
                                        )

                                        if not verification_decision.sql:
                                            log_buffer.append(
                                                f"{partition_id} Verification: ✗ No regenerated SQL provided, using original"
                                            )
                                            # Use original result
                                            filled = sum(
                                                1
                                                for _, row in processed_null_rows.iterrows()
                                                if all(
                                                    pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols
                                                )
                                            )
                                            placeholder = sum(
                                                1
                                                for _, row in processed_null_rows.iterrows()
                                                if any(row[col] == "UNKNOWN" for col in null_cols)
                                            )
                                            discarded = len(null_rows) - len(processed_null_rows)
                                            stats = {
                                                "filled": filled,
                                                "discarded": discarded,
                                                "placeholder": placeholder,
                                            }
                                            log_buffer.append(
                                                f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                            )
                                            log_buffer.append(f"{'=' * 80}\n")
                                            return (
                                                partition_value,
                                                processed_null_rows,
                                                context_rows,
                                                True,
                                                stats,
                                                log_buffer,
                                            )

                                        log_buffer.append(
                                            f"{partition_id} Verification: Reasoning: {verification_decision.reasoning}"
                                        )
                                        log_buffer.append(f"{partition_id} Verification: ORIGINAL SQL:")
                                        log_buffer.append(f"{generated_sql}")
                                        log_buffer.append(f"{partition_id} Verification: REGENERATED SQL:")
                                        log_buffer.append(f"{verification_decision.sql}")

                                        # Execute regenerated SQL on null_rows
                                        try:
                                            new_result = conn.execute(verification_decision.sql).fetchdf()
                                            log_buffer.append(
                                                f"{partition_id} Verification: ✓ Regeneration: {len(null_rows)} → {len(new_result)} rows (auto-accepted)"
                                            )

                                            # Count filled, discarded, placeholder
                                            filled = sum(
                                                1
                                                for _, row in new_result.iterrows()
                                                if all(
                                                    pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols
                                                )
                                            )
                                            placeholder = sum(
                                                1
                                                for _, row in new_result.iterrows()
                                                if any(row[col] == "UNKNOWN" for col in null_cols)
                                            )
                                            discarded = len(null_rows) - len(new_result)

                                            stats = {
                                                "filled": filled,
                                                "discarded": discarded,
                                                "placeholder": placeholder,
                                            }

                                            log_buffer.append(
                                                f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                            )
                                            log_buffer.append(f"{'=' * 80}\n")
                                            return (partition_value, new_result, context_rows, True, stats, log_buffer)
                                        except Exception as e:
                                            log_buffer.append(
                                                f"{partition_id} Verification: ✗ Regenerated SQL failed: {e}"
                                            )
                                            log_buffer.append(
                                                f"{partition_id} Verification: Falling back to original result"
                                            )
                                            # Use original result
                                            filled = sum(
                                                1
                                                for _, row in processed_null_rows.iterrows()
                                                if all(
                                                    pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols
                                                )
                                            )
                                            placeholder = sum(
                                                1
                                                for _, row in processed_null_rows.iterrows()
                                                if any(row[col] == "UNKNOWN" for col in null_cols)
                                            )
                                            discarded = len(null_rows) - len(processed_null_rows)
                                            stats = {
                                                "filled": filled,
                                                "discarded": discarded,
                                                "placeholder": placeholder,
                                            }
                                            log_buffer.append(
                                                f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                            )
                                            log_buffer.append(f"{'=' * 80}\n")
                                            return (
                                                partition_value,
                                                processed_null_rows,
                                                context_rows,
                                                True,
                                                stats,
                                                log_buffer,
                                            )

                                    else:
                                        # Unexpected action
                                        log_buffer.append(
                                            f"{partition_id} Verification: Unexpected action '{verification_decision.action}', auto-approving"
                                        )
                                        filled = sum(
                                            1
                                            for _, row in processed_null_rows.iterrows()
                                            if all(pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols)
                                        )
                                        placeholder = sum(
                                            1
                                            for _, row in processed_null_rows.iterrows()
                                            if any(row[col] == "UNKNOWN" for col in null_cols)
                                        )
                                        discarded = len(null_rows) - len(processed_null_rows)
                                        stats = {"filled": filled, "discarded": discarded, "placeholder": placeholder}
                                        log_buffer.append(
                                            f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                        )
                                        log_buffer.append(f"{'=' * 80}\n")
                                        return (
                                            partition_value,
                                            processed_null_rows,
                                            context_rows,
                                            True,
                                            stats,
                                            log_buffer,
                                        )

                                # Max inspections reached without decision
                                log_buffer.append(
                                    f"{partition_id} Verification: Max inspections reached, auto-approving"
                                )
                                filled = sum(
                                    1
                                    for _, row in processed_null_rows.iterrows()
                                    if all(pd.notna(row[col]) and row[col] != "UNKNOWN" for col in null_cols)
                                )
                                placeholder = sum(
                                    1
                                    for _, row in processed_null_rows.iterrows()
                                    if any(row[col] == "UNKNOWN" for col in null_cols)
                                )
                                discarded = len(null_rows) - len(processed_null_rows)
                                stats = {"filled": filled, "discarded": discarded, "placeholder": placeholder}
                                log_buffer.append(
                                    f"{partition_id} Stats - Filled: {filled}, Discarded: {discarded}, Placeholder: {placeholder}"
                                )
                                log_buffer.append(f"{'=' * 80}\n")
                                return (partition_value, processed_null_rows, context_rows, True, stats, log_buffer)

                            except Exception as e:
                                log_buffer.append(f"{partition_id} ✗ Processing SQL failed: {e}")
                                log_buffer.append(f"{'=' * 80}\n")
                                return (partition_value, null_rows, context_rows, False, {}, log_buffer)
                        else:
                            log_buffer.append(f"{partition_id} No processing SQL provided")
                            log_buffer.append(f"{'=' * 80}\n")
                            return (partition_value, null_rows, context_rows, False, {}, log_buffer)

                # If we exhausted inspections without processing
                log_buffer.append(f"{partition_id} Exhausted inspection budget, keeping NULL rows as-is")
                log_buffer.append(f"{'=' * 80}\n")
                return (partition_value, null_rows, context_rows, False, {}, log_buffer)

            finally:
                conn.close()

        except Exception as e:
            log_buffer.append(f"{partition_id} Partition processing exception: {e}")
            log_buffer.append(f"{'=' * 80}\n")
            return (partition_value, null_rows, context_rows, False, {}, log_buffer)

    async def handle_null_primary_keys_by_document(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        document_column: str = "document_name",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Handle NULL primary keys by processing each document independently in parallel.

        This is a simpler version of the global null handling that processes each document
        separately. Unlike the global version, it doesn't iterate over NULL patterns or
        group by non-NULL PK values within a document.

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names
            metadata: Metadata dictionary for logging
            document_column: Column name containing document identifiers

        Returns:
            Tuple of (processed_dataframe, null_handling_info)
        """
        df = table.dataframe
        table_name = table.dataframe_table_name

        logger.info(f"=== Handling NULL Primary Keys by Document in {table.name} ===")
        logger.info(f"Primary Key: {primary_key}")
        logger.info(f"Initial rows: {len(df)}")

        if document_column not in df.columns:
            logger.warning(f"Document column '{document_column}' not found. Falling back to global null handling.")
            # Fall back to global handling
            return await self.handle_null_primary_keys(table, schema, primary_key, None, metadata)

        # Group by document
        documents = df[document_column].unique()
        logger.info(f"Found {len(documents)} documents to process in parallel")

        # Create tasks for each document
        tasks = []
        document_original_dfs = {}  # Map document_name -> original DataFrame

        for document_name in documents:
            document_df = df[df[document_column] == document_name].copy()
            document_original_dfs[document_name] = document_df

            task = self._handle_single_document(
                document_name=document_name,
                document_df=document_df,
                table_name=table_name,
                schema=schema,
                primary_key=primary_key,
                metadata=metadata,
            )
            tasks.append((document_name, task))

        # Run all document null handling in parallel
        logger.info(f"Starting parallel null handling of {len(tasks)} documents...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Collect results, preserving original rows on exception
        all_processed_dfs = []
        failed_documents = []
        successful_documents = []
        total_null_rows = 0
        total_filled = 0
        total_discarded = 0
        total_placeholder = 0

        for (document_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Document null handling failed for '{document_name}': {type(result).__name__}: {result}")
                logger.error(
                    f"  Preserving original {len(document_original_dfs[document_name])} rows for this document"
                )
                # Preserve original rows on failure
                all_processed_dfs.append(document_original_dfs[document_name])
                failed_documents.append(document_name)
            else:
                _, processed_df, stats = result
                all_processed_dfs.append(processed_df)
                successful_documents.append(document_name)
                total_null_rows += stats.get("null_rows_processed", 0)
                total_filled += stats.get("filled_count", 0)
                total_discarded += stats.get("discarded_count", 0)
                total_placeholder += stats.get("placeholder_count", 0)

        # Concatenate all results
        final_df = pd.concat(all_processed_dfs, ignore_index=True)

        logger.info("\n=== Document-Level NULL PK Handling Complete ===")
        logger.info(f"Documents processed: {len(successful_documents)}/{len(documents)}")
        if failed_documents:
            logger.warning(f"Failed documents: {failed_documents}")
        logger.info(f"Total NULL rows processed: {total_null_rows}")
        logger.info(f"  Filled: {total_filled}")
        logger.info(f"  Discarded: {total_discarded}")
        logger.info(f"  Placeholders: {total_placeholder}")
        logger.info(f"Final rows: {len(final_df)}")

        null_handling_info = {
            "total_null_rows": total_null_rows,
            "filled_count": total_filled,
            "discarded_count": total_discarded,
            "placeholder_count": total_placeholder,
            "initial_rows": len(df),
            "final_rows": len(final_df),
            "documents_processed": len(successful_documents),
            "documents_failed": len(failed_documents),
        }

        return final_df, null_handling_info

    async def _handle_single_document(
        self,
        document_name: str,
        document_df: pd.DataFrame,
        table_name: str,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
    ) -> tuple[str, pd.DataFrame, dict]:
        """
        Handle NULL primary keys for a single document.

        Args:
            document_name: Name of the document
            document_df: DataFrame containing only rows from this document
            table_name: Name of the table
            schema: Schema information
            primary_key: List of primary key field names
            metadata: Metadata dictionary for logging

        Returns:
            Tuple of (document_name, processed_dataframe, stats)
        """
        logger.info(f"\n--- Processing document: {document_name} ---")
        logger.info(f"  Document rows: {len(document_df)}")

        # Partition rows: null rows (any NULL in PK) vs context rows (all PK non-NULL)
        # Build mask for null rows: any PK column is NULL
        null_mask = document_df[primary_key].isna().any(axis=1)
        null_rows = document_df[null_mask]

        # Build mask for context rows: all PK columns non-NULL
        context_mask = document_df[primary_key].notna().all(axis=1)
        context_rows = document_df[context_mask]

        logger.info(f"  NULL rows: {len(null_rows)}")
        logger.info(f"  Context rows: {len(context_rows)}")

        if len(null_rows) == 0:
            logger.info(f"  No NULL rows in document {document_name}, skipping")
            return (
                document_name,
                document_df,
                {
                    "null_rows_processed": 0,
                    "filled_count": 0,
                    "discarded_count": 0,
                    "placeholder_count": 0,
                },
            )

        # Determine which columns are NULL
        # Since this is document-level, we don't need complex pattern iteration
        # We'll just process all null rows together with all context rows
        null_cols = []
        for col in primary_key:
            if null_rows[col].isna().any():
                null_cols.append(col)
            # Note: All context rows are used regardless of which PK columns are non-NULL

        logger.info(f"  NULL columns detected: {null_cols}")

        # Process this document as a single partition
        partition_value = (document_name,)  # Use document name as partition identifier

        try:
            result = await self._process_single_partition(
                null_rows=null_rows,
                context_rows=context_rows,
                null_cols=null_cols,
                non_null_cols=[],  # Not used for document-level
                partition_value=partition_value,
                table_name=table_name,
                schema=schema,
                primary_key=primary_key,
                metadata=metadata,
                mode="document_level",  # Use document-level mode
            )

            partition_value_result, processed_null_rows, _, success, stats, log_buffer = result

            # Print buffered logs
            for log_msg in log_buffer:
                logger.info(log_msg)

            if success:
                # Combine context rows (unchanged) with processed null rows
                remaining_df = document_df[~document_df.index.isin(null_rows.index)]
                if len(processed_null_rows) > 0:
                    final_df = pd.concat([remaining_df, processed_null_rows], ignore_index=True)
                else:
                    final_df = remaining_df

                return (
                    document_name,
                    final_df,
                    {
                        "null_rows_processed": len(null_rows),
                        "filled_count": stats.get("filled", 0),
                        "discarded_count": stats.get("discarded", 0),
                        "placeholder_count": stats.get("placeholder", 0),
                    },
                )
            else:
                logger.warning(f"Document {document_name} processing failed, keeping original rows")
                return (
                    document_name,
                    document_df,
                    {
                        "null_rows_processed": 0,
                        "filled_count": 0,
                        "discarded_count": 0,
                        "placeholder_count": 0,
                    },
                )

        except Exception as e:
            logger.error(f"Exception processing document {document_name}: {e}")
            return (
                document_name,
                document_df,
                {
                    "null_rows_processed": 0,
                    "filled_count": 0,
                    "discarded_count": 0,
                    "placeholder_count": 0,
                },
            )

    async def handle_null_primary_keys_two_pass(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        duck_sql_conn,
        metadata: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Two-pass NULL primary key handling:
        1. Document-level: Handle NULLs within each document independently
        2. Global-level: Handle remaining cross-document NULL patterns

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names
            duck_sql_conn: DuckSQLBasic connection with registered table
            metadata: Metadata dictionary for logging

        Returns:
            Tuple of (processed_dataframe, two_pass_info)
        """
        from sliders.llm_models import ExtractedTable

        logger.info("=" * 80)
        logger.info("TWO-PASS NULL PK HANDLING START")
        logger.info("=" * 80)

        two_pass_info = {
            "pass_1_document_level": {},
            "pass_2_global_level": {},
        }

        # === PASS 1: Document-level NULL handling ===
        logger.info("\n=== PASS 1: DOCUMENT-LEVEL NULL HANDLING ===")
        doc_handled_df, doc_level_info = await self.handle_null_primary_keys_by_document(
            table, schema, primary_key, metadata
        )
        two_pass_info["pass_1_document_level"] = doc_level_info

        # === PASS 2: Global NULL handling ===
        logger.info("\n=== PASS 2: GLOBAL NULL HANDLING ===")

        # Create new ExtractedTable with document-handled data
        global_table = ExtractedTable(
            name=table.name,
            tables=schema,
            sql_query=None,
            dataframe=doc_handled_df,
            dataframe_table_name=table.dataframe_table_name,
            table_str=str(doc_handled_df),
        )

        # Re-register the table with document-handled data
        duck_sql_conn.register(
            doc_handled_df,
            table.dataframe_table_name,
            schema=schema,
            schema_table_name=table.name,
        )

        final_df, global_level_info = await self.handle_null_primary_keys(
            global_table, schema, primary_key, duck_sql_conn, metadata
        )
        two_pass_info["pass_2_global_level"] = global_level_info

        logger.info("=" * 80)
        logger.info("TWO-PASS NULL PK HANDLING COMPLETE")
        logger.info("=" * 80)

        return final_df, two_pass_info

    async def handle_null_non_pk_columns_by_document(
        self,
        table,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
        document_column: str = "document_name",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Handle NULL values in non-primary-key columns by processing each document independently.

        This processes all rows from each document together to fill gaps in data columns
        by broadcasting non-NULL values within the same document.

        Args:
            table: Table object with dataframe
            schema: Schema information
            primary_key: List of primary key field names (to identify non-PK columns)
            metadata: Metadata dictionary for logging
            document_column: Column name containing document identifiers

        Returns:
            Tuple of (processed_dataframe, non_pk_null_handling_info)
        """
        df = table.dataframe
        table_name = table.dataframe_table_name

        logger.info(f"=== Handling NULL Non-PK Columns by Document in {table.name} ===")
        logger.info(f"Primary Key: {primary_key}")
        logger.info(f"Initial rows: {len(df)}")

        if document_column not in df.columns:
            logger.warning(f"Document column '{document_column}' not found. Skipping non-PK null handling.")
            return df, {
                "documents_processed": 0,
                "rows_modified": 0,
                "initial_rows": len(df),
                "final_rows": len(df),
            }

        # Group by document
        documents = df[document_column].unique()
        logger.info(f"Found {len(documents)} documents to process in parallel")

        # Create tasks for each document
        tasks = []
        document_original_dfs = {}

        for document_name in documents:
            document_df = df[df[document_column] == document_name].copy()
            document_original_dfs[document_name] = document_df

            task = self._handle_single_document_non_pk(
                document_name=document_name,
                document_df=document_df,
                table_name=table_name,
                schema=schema,
                primary_key=primary_key,
                metadata=metadata,
            )
            tasks.append((document_name, task))

        # Run all document processing in parallel
        logger.info(f"Starting parallel non-PK null handling of {len(tasks)} documents...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Collect results
        all_processed_dfs = []
        failed_documents = []
        successful_documents = []
        total_rows_modified = 0

        for (document_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Document non-PK null handling failed for '{document_name}': {type(result).__name__}: {result}"
                )
                logger.error(
                    f"  Preserving original {len(document_original_dfs[document_name])} rows for this document"
                )
                # Preserve original rows on failure
                all_processed_dfs.append(document_original_dfs[document_name])
                failed_documents.append(document_name)
            else:
                _, processed_df, stats = result
                all_processed_dfs.append(processed_df)
                successful_documents.append(document_name)
                total_rows_modified += stats.get("rows_modified", 0)

        # Concatenate all results
        final_df = pd.concat(all_processed_dfs, ignore_index=True)

        logger.info("\n=== Document-Level Non-PK NULL Handling Complete ===")
        logger.info(f"Documents processed: {len(successful_documents)}/{len(documents)}")
        if failed_documents:
            logger.warning(f"Failed documents: {failed_documents}")
        logger.info(f"Rows modified: {total_rows_modified}")
        logger.info(f"Final rows: {len(final_df)}")

        non_pk_null_info = {
            "documents_processed": len(successful_documents),
            "documents_failed": len(failed_documents),
            "rows_modified": total_rows_modified,
            "initial_rows": len(df),
            "final_rows": len(final_df),
        }

        return final_df, non_pk_null_info

    async def _handle_single_document_non_pk(
        self,
        document_name: str,
        document_df: pd.DataFrame,
        table_name: str,
        schema: Tables,
        primary_key: list[str],
        metadata: dict,
    ) -> tuple[str, pd.DataFrame, dict]:
        """
        Handle NULL non-PK columns for a single document.

        Args:
            document_name: Name of the document
            document_df: DataFrame containing only rows from this document
            table_name: Name of the table
            schema: Schema information
            primary_key: List of primary key field names
            metadata: Metadata dictionary for logging

        Returns:
            Tuple of (document_name, processed_dataframe, stats)
        """
        logger.info(f"\n--- Processing non-PK nulls for document: {document_name} ---")
        logger.info(f"  Document rows: {len(document_df)}")

        # Check if there are any NULLs in non-PK columns
        non_pk_cols = [col for col in document_df.columns if col not in primary_key]
        has_nulls = document_df[non_pk_cols].isna().any().any()

        if not has_nulls:
            logger.info(f"  No NULLs in non-PK columns for document {document_name}, skipping")
            return (
                document_name,
                document_df,
                {
                    "rows_modified": 0,
                },
            )

        # Create temporary DuckDB connection for this document
        conn = duckdb.connect()
        try:
            # Register the document rows
            conn.register("document_rows", document_df)

            # Initialize LLM chain
            model = self.model_config.get("handle_null_pks", {}).get("model", "gpt-4.1")
            llm_client = get_llm_client(model=model, temperature=0.0)

            non_pk_template = load_fewshot_prompt_template(
                template_file="sliders/handle_null_non_pk_columns.prompt",
                template_blocks=[],
            )
            non_pk_chain = non_pk_template | llm_client.with_structured_output(
                NullPKDecision, method="function_calling"
            )

            # Inspection loop
            max_inspections = self.null_pk_config.get("max_inspections", 5)
            inspection_history = []

            for inspection_num in range(max_inspections + 1):
                inspections_remaining = max_inspections - inspection_num
                must_process_now = inspection_num == max_inspections

                handler = LoggingHandler(
                    prompt_file="sliders/handle_null_non_pk_columns.prompt",
                    metadata={
                        "table_name": table_name,
                        "document_name": document_name,
                        "stage": f"non_pk_null_doc_{document_name}_inspection_{inspection_num}",
                        "question_id": metadata.get("question_id", None),
                        **(metadata or {}),
                    },
                )

                # Format inspection history
                formatted_history = "\n".join(
                    [
                        f"Inspection {i + 1}:\nSQL: {sql}\nResult:\n{result}\n"
                        for i, (sql, result) in enumerate(inspection_history)
                    ]
                )

                # Get table stats
                document_stats = get_table_stats(document_df, "document_rows")

                decision = await non_pk_chain.ainvoke(
                    {
                        "table_name": table_name,
                        "schema": format_table_schema(schema, table_name),
                        "primary_key": primary_key,
                        "document_name": document_name,
                        "row_count": len(document_df),
                        "document_rows_stats": format_table_stats(document_stats),
                        "inspection_history": formatted_history if formatted_history else "No inspections yet",
                        "inspections_remaining": inspections_remaining,
                        "must_process_now": must_process_now,
                    },
                    config={"callbacks": [handler]},
                )

                if decision.action == "inspect":
                    if decision.sql and inspections_remaining > 0:
                        logger.info(f"  Inspection {inspection_num + 1}/{max_inspections}: {decision.sql[:100]}...")
                        try:
                            result = conn.execute(decision.sql).fetchdf()
                            formatted_result = format_sql_result(result)
                            inspection_history.append((decision.sql, formatted_result))
                            logger.info("  ✓ Inspection successful")
                        except Exception as e:
                            error_msg = f"SQL Error: {str(e)}"
                            logger.info(f"  ✗ Inspection failed: {e}")
                            inspection_history.append((decision.sql, error_msg))
                    else:
                        logger.info("  No SQL provided or inspection budget exhausted")
                        break

                elif decision.action == "process":
                    logger.info("  Agent ready to process non-PK NULLs")
                    logger.info(f"  Reasoning: {decision.reasoning[:200]}...")

                    # Execute processing SQL
                    if decision.sql:
                        logger.info(f"  Processing SQL: {decision.sql[:200]}...")
                        try:
                            processed_df = conn.execute(decision.sql).fetchdf()
                            logger.info(
                                f"  ✓ Processing successful: {len(document_df)} rows → {len(processed_df)} rows"
                            )

                            # Count how many rows were modified (had NULLs filled)
                            rows_modified = 0
                            if len(processed_df) == len(document_df):
                                # Compare original vs processed to count modifications
                                for col in non_pk_cols:
                                    if col in document_df.columns and col in processed_df.columns:
                                        orig_nulls = document_df[col].isna()
                                        proc_nulls = processed_df[col].isna()
                                        filled = (orig_nulls & ~proc_nulls).sum()
                                        rows_modified += filled

                            stats = {"rows_modified": rows_modified}

                            logger.info(f"  Rows with filled NULLs: {rows_modified}")
                            return (document_name, processed_df, stats)

                        except Exception as e:
                            logger.error(f"  ✗ Processing SQL failed: {e}")
                            return (document_name, document_df, {"rows_modified": 0})
                    else:
                        logger.warning("  No processing SQL provided")
                        return (document_name, document_df, {"rows_modified": 0})

            # If we exhausted inspections without processing
            logger.info("  Exhausted inspection budget, keeping rows as-is")
            return (document_name, document_df, {"rows_modified": 0})

        except Exception as e:
            logger.error(f"  Exception processing document {document_name}: {e}")
            return (
                document_name,
                document_df,
                {
                    "rows_modified": 0,
                },
            )
        finally:
            conn.close()
