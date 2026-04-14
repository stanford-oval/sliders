"""
Reconciliation Pipeline V2

Simplified architecture that processes each primary key group independently:
1. PK Selection → 2. PK Canonicalization → 2.5 NULL PK Handling → 3. Per-PK Controller-Executor Loops (parallel)

For each PK group with multiple rows:
- Controller inspects and chooses ONE operation at a time
- Executor performs that operation
- Loop continues (controller → executor) up to 5 iterations
- Controller can choose: deduplicate, aggregate, resolve_conflicts, canonicalize, or stop

Reconciliation Transparency:
- All reconciled rows include a __reconciliation_context__ field (TEXT/JSON)
- This field documents what reconciliation was performed and what information was discarded
- Ensures downstream users have complete visibility into reconciliation decisions
"""

import pandas as pd
import asyncio
import json
from typing import Optional, Literal
from pydantic import BaseModel
from pathlib import Path
from copy import deepcopy

from sliders.document import Document
from sliders.llm_models import Tables, ExtractedTable
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.utils import (
    get_table_stats,
    format_table_stats,
    format_sql_result,
    format_dataframe_schema,
    run_sql,
)
from sliders.modules.primary_key_selector import PrimaryKeySelector
from sliders.modules.canonicalize_fields import FieldCanonicalizer
from sliders.modules.handle_null_primary_keys import NullPKHandler
from sliders.llm_tools import DuckSQLBasic
import duckdb


# ============================================================================
# Pydantic Models
# ============================================================================


class ControllerDecisionV2(BaseModel):
    """Decision from controller for a single PK group."""

    reasoning: str
    action: Literal["inspect", "route"]
    sql: Optional[str] = None  # Inspection SQL if action is "inspect"
    route_to: Optional[
        Literal["deduplicate", "aggregate", "consolidate", "resolve_conflicts", "canonicalize", "stop"]
    ] = None


class ExecutorDecisionV2(BaseModel):
    """Decision from executor for a single PK group."""

    reasoning: str
    action: Literal["inspect", "generate_merge_sql", "approve", "regenerate"]
    sql: Optional[str] = None  # Inspection SQL, merge SQL, regenerated SQL, or None for approve


class ReconciliationContextDecision(BaseModel):
    """Decision from reconciliation context generator."""

    reasoning: str
    context_json: str  # JSON string with reconciliation context


class ColumnSelectorDecision(BaseModel):
    """Decision from column selector for non-PK canonicalization."""

    reasoning: str
    action: Literal["inspect", "finalize"]
    sql: Optional[str] = None  # Inspection SQL if action is "inspect"
    columns_to_canonicalize: Optional[list[str]] = None  # Final list when action="finalize"


class PKGroupState:
    """State tracking for a single PK group's controller-executor loop."""

    def __init__(self, pk_value: tuple, initial_df: pd.DataFrame):
        self.pk_value = pk_value
        self.current_df = initial_df
        self.operations_performed: list[str] = []
        self.iteration = 0
        self.controller_inspections: list[dict] = []
        self.executor_results: list[dict] = []


# ============================================================================
# Core V2 Functions
# ============================================================================


def split_by_pk_groups(df: pd.DataFrame, primary_key: list[str]) -> dict[tuple, pd.DataFrame]:
    """
    Split table into groups based on primary key values.

    Returns:
        dict mapping PK value tuples to DataFrames containing rows with that PK
    """
    if not primary_key:
        # No primary key - treat entire table as one group
        return {tuple(): df}

    # Group by primary key
    pk_groups = {}
    for pk_value, group_df in df.groupby(primary_key, dropna=False):
        # Ensure pk_value is always a tuple (even for single-column PKs)
        if not isinstance(pk_value, tuple):
            pk_value = (pk_value,)
        pk_groups[pk_value] = group_df.reset_index(drop=True)

    return pk_groups


async def generate_reconciliation_context(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    sql_executed: str,
    executor_reasoning: str,
    operation: str,
    pk_value: tuple,
    primary_key: list[str],
    metadata: dict,
    log_buffer: list[str],
    partition_id: str,
    model: str = "gpt-4.1",
    max_rows: int = 20,
) -> str:
    """
    Generate reconciliation context JSON string documenting what happened during reconciliation.

    Args:
        initial_df: DataFrame before reconciliation
        final_df: DataFrame after reconciliation
        sql_executed: SQL that was executed
        executor_reasoning: Reasoning from the executor agent
        operation: Operation type (deduplicate, aggregate, resolve_conflicts, canonicalize)
        pk_value: Primary key value being processed
        primary_key: List of primary key column names
        metadata: Metadata dict for logging
        log_buffer: Log buffer for this partition
        partition_id: Partition ID for logging
        model: Model name to use for LLM client
        max_rows: Maximum number of rows to show in context

    Returns:
        JSON string with reconciliation context
    """
    try:
        log_buffer.append(f"{partition_id} Context Generator: Starting")

        # Get LLM client with the same model used for reconciliation
        llm_client = get_llm_client(model=model)

        # Load prompt template
        context_template = load_fewshot_prompt_template(
            template_file="sliders/reconcilation/context_generator.prompt",
            template_blocks=[],
        )

        # Create chain with structured output
        context_chain = context_template | llm_client.with_structured_output(ReconciliationContextDecision)

        # Format data for prompt
        initial_data = format_sql_result(initial_df, max_rows=max_rows)
        final_data = format_sql_result(final_df, max_rows=max_rows) if len(final_df) > 0 else "Empty result"

        handler = LoggingHandler(
            prompt_file="sliders/reconcilation/context_generator.prompt",
            metadata={
                "stage": f"context_generator_pk_{pk_value}_op_{operation}",
                **(metadata or {}),
            },
        )

        # Call LLM
        decision = await context_chain.ainvoke(
            {
                "operation": operation,
                "primary_key": primary_key,
                "pk_value": pk_value,
                "initial_row_count": len(initial_df),
                "final_row_count": len(final_df),
                "initial_schema": format_dataframe_schema(initial_df),
                "final_schema": format_dataframe_schema(final_df) if len(final_df) > 0 else "Empty",
                "initial_data": initial_data,
                "final_data": final_data,
                "sql_executed": sql_executed,
                "executor_reasoning": executor_reasoning,
            },
            config={"callbacks": [handler]},
        )

        log_buffer.append(f"{partition_id} Context Generator: ✓ Generated context")
        return decision.context_json

    except Exception as e:
        # Fallback context if generation fails
        log_buffer.append(f"{partition_id} Context Generator: ✗ Failed: {e}")
        fallback_context = {
            "operation": operation,
            "original_row_count": len(initial_df),
            "final_row_count": len(final_df),
            "summary": f"Reconciliation performed ({operation}), context generation failed",
            "information_discarded": f"unknown - context generation error: {str(e)[:100]}",
        }
        return json.dumps(fallback_context)


async def select_columns_for_canonicalization(
    table_df: pd.DataFrame,
    primary_key: list[str],
    table_name: str,
    schema: Tables,
    metadata: dict,
    model_config: dict,
    column_selector_config: dict = None,
) -> list[str]:
    """
    Use LLM agent to identify non-PK columns that need canonicalization.

    Args:
        table_df: DataFrame to analyze
        primary_key: List of primary key columns (to exclude)
        table_name: Name of the table
        schema: Table schema
        metadata: Metadata dict for logging
        model_config: Model configuration
        column_selector_config: Column selector configuration

    Returns:
        List of column names to canonicalize
    """
    logger.info("=== Column Selection for Non-PK Canonicalization ===")
    logger.info(f"Table: {table_name}")
    logger.info(f"Total columns: {len(table_df.columns)}")

    # Extract column selector config
    column_selector_config = column_selector_config or {}
    max_inspections = column_selector_config.get("max_inspections", 5)
    excluded_column_names = column_selector_config.get(
        "excluded_columns", ["row_id", "page_number", "__reconciliation_context__", "number_instances"]
    )
    excluded_patterns = column_selector_config.get("excluded_patterns", ["_quote", "_rationale", "_confidence"])

    # Initialize LLM chain
    model = model_config.get("column_selector", {}).get("model", "gpt-4.1")
    llm_client = get_llm_client(model=model)

    selector_template = load_fewshot_prompt_template(
        template_file="sliders/reconcilation/column_selector_for_canonicalization.prompt",
        template_blocks=[],
    )
    selector_chain = selector_template | llm_client.with_structured_output(ColumnSelectorDecision)

    # Automatically exclude certain columns
    excluded_columns = set()

    # Exclude primary key columns
    excluded_columns.update(primary_key)

    # Exclude configured system columns
    excluded_columns.update(excluded_column_names)

    # Exclude metadata columns (patterns)
    for col in table_df.columns:
        for pattern in excluded_patterns:
            if col.endswith(pattern):
                excluded_columns.add(col)
                break

    logger.info(f"Automatically excluded columns: {sorted(excluded_columns)}")

    # Get table stats
    table_stats = get_table_stats(table_df, table_name)
    inspection_history = []

    # Inspection loop
    for i in range(max_inspections + 1):  # +1 for final decision
        inspections_remaining = max_inspections - i

        handler = LoggingHandler(
            prompt_file="sliders/reconcilation/column_selector_for_canonicalization.prompt",
            metadata={
                "stage": f"column_selector_inspection_{i}",
                **(metadata or {}),
            },
        )

        # Format inspection history
        formatted_inspections = []
        for idx, (sql, result) in enumerate(inspection_history):
            formatted_inspections.append(f"Inspection {idx + 1}:\nSQL: {sql}\nResult:\n{result}\n")

        decision = await selector_chain.ainvoke(
            {
                "table_name": table_name,
                "primary_key": primary_key,
                "schema": schema,
                "table_stats": format_table_stats(table_stats),
                "inspection_history": "\n".join(formatted_inspections)
                if formatted_inspections
                else "No inspections yet",
                "inspections_remaining": inspections_remaining,
                "max_inspections": max_inspections,
            },
            config={"callbacks": [handler]},
        )

        logger.info(f"Column Selector (inspection {i}): Action = {decision.action}")
        logger.info(f"Reasoning: {decision.reasoning}")

        if decision.action == "inspect":
            # Execute inspection SQL
            try:
                logger.info(f"Executing inspection SQL: {decision.sql}")
                result = run_sql(decision.sql, table_df, table_name)
                formatted_result = format_sql_result(result)
                inspection_history.append((decision.sql, formatted_result))
                logger.info(f"✓ Inspection {i + 1} successful")
            except Exception as e:
                error_msg = f"SQL Error: {str(e)}"
                logger.warning(f"✗ Inspection {i + 1} failed: {e}")
                inspection_history.append((decision.sql, error_msg))

        elif decision.action == "finalize":
            # Agent provided final column list
            columns_to_canonicalize = decision.columns_to_canonicalize or []

            # Filter out any excluded columns (safety check)
            filtered_columns = [col for col in columns_to_canonicalize if col not in excluded_columns]

            if len(filtered_columns) < len(columns_to_canonicalize):
                removed = set(columns_to_canonicalize) - set(filtered_columns)
                logger.warning(f"Removed excluded columns from selection: {removed}")

            logger.info(f"✓ Column selection complete: {len(filtered_columns)} columns selected")
            if filtered_columns:
                logger.info(f"  Columns: {filtered_columns}")
            else:
                logger.info("  No columns need canonicalization")

            return filtered_columns

    # Fallback if no finalization (should not happen)
    logger.warning("✗ Column selector never finalized, returning empty list")
    return []


async def run_controller_for_pk(
    current_df: pd.DataFrame,
    primary_key: list[str],
    pk_value: tuple,
    iteration: int,
    max_iterations: int,
    operations_history: list[str],
    question: str,
    schema: Tables,
    table_name: str,
    controller_chain,
    metadata: dict,
    max_controller_inspections: int = 3,
    log_buffer: list[str] = None,
    partition_id: str = "",
) -> tuple[str, list[str]]:
    """
    Run controller agent for a single PK group (one call).

    Controller can inspect (up to max_controller_inspections times) then must route to ONE operation.

    Returns:
        Tuple of (operation name, log_buffer)
    """
    if log_buffer is None:
        log_buffer = []

    inspection_history = []
    route_to = None

    # Get table stats for this PK group
    table_stats = get_table_stats(current_df, table_name)

    for i in range(max_controller_inspections + 1):  # +1 for final routing
        inspections_remaining = max_controller_inspections - i
        must_route_now = i == max_controller_inspections

        handler = LoggingHandler(
            prompt_file="sliders/reconcilation/controller_v2.prompt",
            metadata={
                "question": question,
                "stage": f"controller_v2_pk_{pk_value}_iter_{iteration}_insp_{i}",
                **(metadata or {}),
            },
        )

        # Format inspection history
        formatted_inspections = []
        for idx, (sql, result) in enumerate(inspection_history):
            formatted_inspections.append(f"Inspection {idx + 1}:\nSQL: {sql}\nResult:\n{result}\n")

        # Format operations history
        ops_history_str = ", ".join(operations_history) if operations_history else "None"

        decision = await controller_chain.ainvoke(
            {
                "question": question,
                "schema": schema,
                "primary_key": primary_key,
                "pk_value": pk_value,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "operations_history": ops_history_str,
                "current_row_count": len(current_df),
                "table_stats": format_table_stats(table_stats),
                "inspection_history": "\n".join(formatted_inspections)
                if formatted_inspections
                else "No inspections yet",
                "table_name": table_name,
                "inspections_remaining": inspections_remaining,
                "must_route_now": must_route_now,
            },
            config={"callbacks": [handler]},
        )

        if decision.action == "inspect":
            # Execute inspection SQL
            try:
                log_buffer.append(f"{partition_id} Controller (iter={iteration}): Inspection {i + 1} SQL")
                log_buffer.append(f"{partition_id} SQL: {decision.sql}")
                result = run_sql(decision.sql, current_df, table_name)
                formatted_result = format_sql_result(result)
                inspection_history.append((decision.sql, formatted_result))
            except Exception as e:
                error_msg = f"SQL Error: {str(e)}"
                log_buffer.append(f"{partition_id} ✗ Controller inspection failed: {e}")
                inspection_history.append((decision.sql, error_msg))

        elif decision.action == "route":
            route_to = decision.route_to
            log_buffer.append(f"{partition_id} Controller (iter={iteration}): Routing to '{route_to}'")
            log_buffer.append(f"{partition_id} Reasoning: {decision.reasoning}")
            break

    # Fallback if no routing decision
    if route_to is None:
        route_to = "stop"
        log_buffer.append(f"{partition_id} ✗ Controller (iter={iteration}): No routing decision, defaulting to 'stop'")

    return route_to, log_buffer


async def run_executor_for_pk(
    current_df: pd.DataFrame,
    operation: str,
    primary_key: list[str],
    pk_value: tuple,
    question: str,
    schema: Tables,
    table_name: str,
    executor_chain,
    metadata: dict,
    model: str = "gpt-4.1",
    controller_reasoning: str = "",
    max_executor_inspections: int = 3,
    max_sql_attempts: int = 3,
    verification_config: dict = None,
    context_generator_config: dict = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run executor agent for a single PK group with assigned operation.

    Executor can inspect (up to max_executor_inspections times), then generates merge SQL (up to max_sql_attempts attempts).

    Returns:
        Tuple of (Updated DataFrame, log_buffer)
    """
    # Buffer log messages for this executor
    log_buffer = []
    partition_id = f"[PK {pk_value}]"

    # Extract context generator config
    context_generator_config = context_generator_config or {}
    enable_context_generator = context_generator_config.get("enable", True)
    context_generator_model = context_generator_config.get("model", model)
    context_generator_max_rows = context_generator_config.get("max_rows", 20)

    inspection_history = []

    # Get table stats
    table_stats = get_table_stats(current_df, table_name)

    # Phase 1: Inspection with optional early SQL generation
    sql_error_feedback = ""
    merge_sql_decision = None

    for i in range(max_executor_inspections + 1):  # +1 for final decision if all inspections used
        inspections_remaining = max(0, max_executor_inspections - i)

        handler = LoggingHandler(
            prompt_file="sliders/reconcilation/executor_v2.prompt",
            metadata={
                "question": question,
                "stage": f"executor_v2_pk_{pk_value}_op_{operation}_step_{i}",
                **(metadata or {}),
            },
        )

        # Format inspection history
        formatted_inspections = []
        for idx, (sql, result) in enumerate(inspection_history):
            formatted_inspections.append(f"Inspection {idx + 1}:\nSQL: {sql}\nResult:\n{result}\n")

        decision = await executor_chain.ainvoke(
            {
                "question": question,
                "schema": schema,
                "primary_key": primary_key,
                "pk_value": pk_value,
                "objective": operation,
                "table_stats": format_table_stats(table_stats),
                "controller_reasoning": controller_reasoning,
                "inspection_history": "\n".join(formatted_inspections)
                if formatted_inspections
                else "No inspections yet",
                "inspection_error_feedback": "",
                "sql_error_feedback": sql_error_feedback,
                "table_name": table_name,
                "inspections_remaining": inspections_remaining,
                # Verification variables (not used in generation phase but required by template)
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
            # Execute inspection SQL
            try:
                log_buffer.append(f"{partition_id} Executor (op={operation}): Inspection {i + 1}")
                log_buffer.append(f"{partition_id} Reasoning: {decision.reasoning}")
                log_buffer.append(f"{partition_id} SQL: {decision.sql}")
                result = run_sql(decision.sql, current_df, table_name)
                formatted_result = format_sql_result(result)
                inspection_history.append((decision.sql, formatted_result))
            except Exception as e:
                error_msg = f"SQL Error: {str(e)}"
                log_buffer.append(f"{partition_id} ✗ Executor inspection failed: {e}")
                inspection_history.append((decision.sql, error_msg))

        elif decision.action == "generate_merge_sql":
            # Executor provided the final merge SQL - this is the intended behavior per prompt
            merge_sql_decision = decision
            log_buffer.append(f"{partition_id} Executor (op={operation}): Generated merge SQL")
            log_buffer.append(f"{partition_id} Reasoning: {decision.reasoning}")
            log_buffer.append(f"{partition_id} SQL: {decision.sql}")
            break

    # If we never got a merge SQL decision, something went wrong
    if merge_sql_decision is None:
        log_buffer.append(
            f"{partition_id} ✗ Executor never produced merge SQL after {max_executor_inspections + 1} calls"
        )
        return current_df, log_buffer

    # Phase 2: Try to execute the merge SQL with retry on failure
    for attempt in range(max_sql_attempts):
        # Try to execute the merge SQL
        try:
            log_buffer.append(
                f"{partition_id} Executor (op={operation}): Executing merge SQL (attempt {attempt + 1}/{max_sql_attempts})"
            )
            result_df = run_sql(merge_sql_decision.sql, current_df, table_name)
            log_buffer.append(f"{partition_id} Executor result: {len(current_df)} rows → {len(result_df)} rows")

            # Success! Now check if verification is enabled
            verification_config = verification_config or {}
            enable_verification = verification_config.get("enable", False)

            if not enable_verification:
                # Skip verification, generate context and return result immediately
                log_buffer.append(f"{partition_id} Verification disabled, generating context")
                if len(result_df) == 0:
                    return pd.DataFrame(), log_buffer

                # Generate reconciliation context if enabled
                if enable_context_generator:
                    context_json = await generate_reconciliation_context(
                        initial_df=current_df,
                        final_df=result_df,
                        sql_executed=merge_sql_decision.sql,
                        executor_reasoning=merge_sql_decision.reasoning,
                        operation=operation,
                        pk_value=pk_value,
                        primary_key=primary_key,
                        metadata=metadata,
                        log_buffer=log_buffer,
                        partition_id=partition_id,
                        model=context_generator_model,
                        max_rows=context_generator_max_rows,
                    )
                    result_df["__reconciliation_context__"] = context_json
                else:
                    # Add minimal context if generator is disabled
                    minimal_context = {
                        "operation": operation,
                        "original_row_count": len(current_df),
                        "final_row_count": len(result_df),
                        "summary": f"Operation: {operation} (context generation disabled)",
                        "information_discarded": "Context generation disabled",
                    }
                    result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                return result_df, log_buffer

            # VERIFICATION PHASE
            log_buffer.append(f"{partition_id} Verification phase: Entering verification")
            initial_table = current_df.copy()
            final_table = result_df
            generated_sql = merge_sql_decision.sql
            generation_reasoning = merge_sql_decision.reasoning

            max_inspections = verification_config.get("max_inspections", 5)
            verification_inspection_count = 0
            verification_inspection_history = []

            # Create connection once for all verification inspections
            conn = duckdb.connect()
            try:
                conn.register("initial_table", initial_table)
                conn.register("final_table", final_table)

                while verification_inspection_count < max_inspections:
                    # Format verification inspection history
                    formatted_verif_inspections = []
                    for idx, insp in enumerate(verification_inspection_history):
                        formatted_verif_inspections.append(
                            f"Inspection {idx + 1}:\nReasoning: {insp['reasoning']}\nSQL: {insp['query']}\nResult:\n{insp['result']}\n"
                        )

                    handler = LoggingHandler(
                        prompt_file="sliders/reconcilation/executor_v2.prompt",
                        metadata={
                            "question": question,
                            "stage": f"executor_v2_pk_{pk_value}_op_{operation}_verification_{verification_inspection_count}",
                            **(metadata or {}),
                        },
                    )

                    verification_decision = await executor_chain.ainvoke(
                        {
                            "pk_value": pk_value,
                            "initial_schema": format_dataframe_schema(initial_table),
                            "final_schema": format_dataframe_schema(final_table),
                            "verification_mode": True,
                            "generated_sql": generated_sql,
                            "generation_reasoning": generation_reasoning,
                            "initial_row_count": len(initial_table),
                            "final_row_count": len(final_table),
                            "remaining_inspections": max_inspections - verification_inspection_count,
                            "verification_inspection_history": "\n".join(formatted_verif_inspections)
                            if formatted_verif_inspections
                            else "No verification inspections yet",
                            "objective": operation,
                            # Regular mode variables (not used in verification but required by template)
                            "question": question,
                            "schema": schema,
                            "primary_key": primary_key,
                            "table_stats": "",
                            "controller_reasoning": "",
                            "table_name": table_name,
                            "inspections_remaining": 0,
                            "inspection_history": "",
                            "inspection_error_feedback": "",
                            "sql_error_feedback": "",
                        },
                        config={"callbacks": [handler]},
                    )

                    if verification_decision.action == "inspect":
                        # Run inspection query on both tables
                        try:
                            log_buffer.append(
                                f"{partition_id} Verification: Inspection {verification_inspection_count + 1}/{max_inspections}"
                            )
                            log_buffer.append(
                                f"{partition_id} Verification: Reasoning: {verification_decision.reasoning}"
                            )
                            log_buffer.append(f"{partition_id} Verification: SQL: {verification_decision.sql}")

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
                        # Agent approved the final_table - generate context
                        log_buffer.append(
                            f"{partition_id} Verification: ✓ Approved after {verification_inspection_count} inspection(s)"
                        )
                        log_buffer.append(f"{partition_id} Verification: {verification_decision.reasoning}")
                        if len(result_df) == 0:
                            return pd.DataFrame(), log_buffer

                        # Generate reconciliation context if enabled
                        if enable_context_generator:
                            context_json = await generate_reconciliation_context(
                                initial_df=initial_table,
                                final_df=result_df,
                                sql_executed=generated_sql,
                                executor_reasoning=generation_reasoning,
                                operation=operation,
                                pk_value=pk_value,
                                primary_key=primary_key,
                                metadata=metadata,
                                log_buffer=log_buffer,
                                partition_id=partition_id,
                                model=context_generator_model,
                                max_rows=context_generator_max_rows,
                            )
                            result_df["__reconciliation_context__"] = context_json
                        else:
                            # Add minimal context if generator is disabled
                            minimal_context = {
                                "operation": operation,
                                "original_row_count": len(initial_table),
                                "final_row_count": len(result_df),
                                "summary": f"Operation: {operation} (context generation disabled)",
                                "information_discarded": "Context generation disabled",
                            }
                            result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                        return result_df, log_buffer

                    elif verification_decision.action == "regenerate":
                        # Agent chose to regenerate and provided new SQL directly
                        log_buffer.append(
                            f"{partition_id} Verification: Regenerating after {verification_inspection_count} inspection(s)"
                        )

                        if not verification_decision.sql:
                            log_buffer.append(
                                f"{partition_id} Verification: ✗ No regenerated SQL provided, using original"
                            )
                            if len(result_df) == 0:
                                return pd.DataFrame(), log_buffer

                            # Generate context for original result if enabled
                            if enable_context_generator:
                                context_json = await generate_reconciliation_context(
                                    initial_df=initial_table,
                                    final_df=result_df,
                                    sql_executed=generated_sql,
                                    executor_reasoning=generation_reasoning,
                                    operation=operation,
                                    pk_value=pk_value,
                                    primary_key=primary_key,
                                    metadata=metadata,
                                    log_buffer=log_buffer,
                                    partition_id=partition_id,
                                    model=context_generator_model,
                                    max_rows=context_generator_max_rows,
                                )
                                result_df["__reconciliation_context__"] = context_json
                            else:
                                # Add minimal context if generator is disabled
                                minimal_context = {
                                    "operation": operation,
                                    "original_row_count": len(initial_table),
                                    "final_row_count": len(result_df),
                                    "summary": f"Operation: {operation} (context generation disabled)",
                                    "information_discarded": "Context generation disabled",
                                }
                                result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                            return result_df, log_buffer

                        log_buffer.append(f"{partition_id} Verification: Reasoning: {verification_decision.reasoning}")
                        log_buffer.append(f"{partition_id} Verification: ORIGINAL SQL:")
                        log_buffer.append(f"{generated_sql}")
                        log_buffer.append(f"{partition_id} Verification: REGENERATED SQL:")
                        log_buffer.append(f"{verification_decision.sql}")

                        # Execute new SQL on initial table (using existing connection with initial_table registered)
                        try:
                            new_result_df = conn.execute(verification_decision.sql).fetchdf()
                            log_buffer.append(
                                f"{partition_id} Verification: ✓ Regeneration: {len(initial_table)} rows → {len(new_result_df)} rows (auto-accepted)"
                            )

                            if len(new_result_df) == 0:
                                return pd.DataFrame(), log_buffer

                            # Generate reconciliation context for regenerated SQL if enabled
                            if enable_context_generator:
                                context_json = await generate_reconciliation_context(
                                    initial_df=initial_table,
                                    final_df=new_result_df,
                                    sql_executed=verification_decision.sql,
                                    executor_reasoning=verification_decision.reasoning,
                                    operation=operation,
                                    pk_value=pk_value,
                                    primary_key=primary_key,
                                    metadata=metadata,
                                    log_buffer=log_buffer,
                                    partition_id=partition_id,
                                    model=context_generator_model,
                                    max_rows=context_generator_max_rows,
                                )
                                new_result_df["__reconciliation_context__"] = context_json
                            else:
                                # Add minimal context if generator is disabled
                                minimal_context = {
                                    "operation": operation,
                                    "original_row_count": len(initial_table),
                                    "final_row_count": len(new_result_df),
                                    "summary": f"Operation: {operation} (context generation disabled)",
                                    "information_discarded": "Context generation disabled",
                                }
                                new_result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                            return new_result_df, log_buffer
                        except Exception as e:
                            log_buffer.append(f"{partition_id} Verification: ✗ Regenerated SQL failed: {e}")
                            log_buffer.append(f"{partition_id} Verification: ORIGINAL SQL:")
                            log_buffer.append(f"{generated_sql}")
                            log_buffer.append(f"{partition_id} Verification: REGENERATED SQL (FAILED):")
                            log_buffer.append(f"{verification_decision.sql}")
                            log_buffer.append(f"{partition_id} Verification: Falling back to original result")
                            if len(result_df) == 0:
                                return pd.DataFrame(), log_buffer

                            # Generate context for original result if enabled
                            if enable_context_generator:
                                context_json = await generate_reconciliation_context(
                                    initial_df=initial_table,
                                    final_df=result_df,
                                    sql_executed=generated_sql,
                                    executor_reasoning=generation_reasoning,
                                    operation=operation,
                                    pk_value=pk_value,
                                    primary_key=primary_key,
                                    metadata=metadata,
                                    log_buffer=log_buffer,
                                    partition_id=partition_id,
                                    model=context_generator_model,
                                    max_rows=context_generator_max_rows,
                                )
                                result_df["__reconciliation_context__"] = context_json
                            else:
                                # Add minimal context if generator is disabled
                                minimal_context = {
                                    "operation": operation,
                                    "original_row_count": len(initial_table),
                                    "final_row_count": len(result_df),
                                    "summary": f"Operation: {operation} (context generation disabled)",
                                    "information_discarded": "Context generation disabled",
                                }
                                result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                            return result_df, log_buffer

                    else:
                        # Unexpected action - log error and generate context before returning
                        log_buffer.append(
                            f"{partition_id} Verification: Unexpected action '{verification_decision.action}', auto-approving"
                        )
                        if len(result_df) == 0:
                            return pd.DataFrame(), log_buffer

                        # Generate reconciliation context if enabled
                        if enable_context_generator:
                            context_json = await generate_reconciliation_context(
                                initial_df=initial_table,
                                final_df=result_df,
                                sql_executed=generated_sql,
                                executor_reasoning=generation_reasoning,
                                operation=operation,
                                pk_value=pk_value,
                                primary_key=primary_key,
                                metadata=metadata,
                                log_buffer=log_buffer,
                                partition_id=partition_id,
                                model=context_generator_model,
                                max_rows=context_generator_max_rows,
                            )
                            result_df["__reconciliation_context__"] = context_json
                        else:
                            # Add minimal context if generator is disabled
                            minimal_context = {
                                "operation": operation,
                                "original_row_count": len(initial_table),
                                "final_row_count": len(result_df),
                                "summary": f"Operation: {operation} (context generation disabled)",
                                "information_discarded": "Context generation disabled",
                            }
                            result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                        return result_df, log_buffer

                # Max inspections reached without decision - approve by default and generate context
                log_buffer.append(
                    f"{partition_id} Verification: Max inspections ({max_inspections}) reached, auto-approving"
                )
                if len(result_df) == 0:
                    return pd.DataFrame(), log_buffer

                # Generate reconciliation context if enabled
                if enable_context_generator:
                    context_json = await generate_reconciliation_context(
                        initial_df=initial_table,
                        final_df=result_df,
                        sql_executed=generated_sql,
                        executor_reasoning=generation_reasoning,
                        operation=operation,
                        pk_value=pk_value,
                        primary_key=primary_key,
                        metadata=metadata,
                        log_buffer=log_buffer,
                        partition_id=partition_id,
                        model=context_generator_model,
                        max_rows=context_generator_max_rows,
                    )
                    result_df["__reconciliation_context__"] = context_json
                else:
                    # Add minimal context if generator is disabled
                    minimal_context = {
                        "operation": operation,
                        "original_row_count": len(initial_table),
                        "final_row_count": len(result_df),
                        "summary": f"Operation: {operation} (context generation disabled)",
                        "information_discarded": "Context generation disabled",
                    }
                    result_df["__reconciliation_context__"] = json.dumps(minimal_context)
                return result_df, log_buffer

            finally:
                # Always close connection, even if exception occurs
                conn.close()

        except Exception as e:
            log_buffer.append(f"{partition_id} ✗ Executor SQL failed (attempt {attempt + 1}/{max_sql_attempts}): {e}")
            log_buffer.append(f"{partition_id} Failed SQL: {merge_sql_decision.sql}")

            # If this wasn't the last attempt, ask for a retry with error feedback
            if attempt < max_sql_attempts - 1:
                sql_error_feedback = f"SQL execution failed on attempt {attempt + 1}:\nError: {str(e)}\nSQL: {merge_sql_decision.sql}\n\nPlease fix the SQL and try again."

                # Make a new LLM call with error feedback
                handler = LoggingHandler(
                    prompt_file="sliders/reconcilation/executor_v2.prompt",
                    metadata={
                        "question": question,
                        "stage": f"executor_v2_pk_{pk_value}_op_{operation}_retry_{attempt + 1}",
                        **(metadata or {}),
                    },
                )

                # Format inspection history
                formatted_inspections = []
                for idx, (sql, result) in enumerate(inspection_history):
                    formatted_inspections.append(f"Inspection {idx + 1}:\nSQL: {sql}\nResult:\n{result}\n")

                merge_sql_decision = await executor_chain.ainvoke(
                    {
                        "question": question,
                        "schema": schema,
                        "primary_key": primary_key,
                        "pk_value": pk_value,
                        "objective": operation,
                        "table_stats": format_table_stats(table_stats),
                        "controller_reasoning": controller_reasoning,
                        "inspection_history": "\n".join(formatted_inspections)
                        if formatted_inspections
                        else "No inspections yet",
                        "inspection_error_feedback": "",
                        "sql_error_feedback": sql_error_feedback,
                        "table_name": table_name,
                        "inspections_remaining": 0,  # No more inspections during retry
                        # Verification variables (not used in retry phase but required by template)
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

                log_buffer.append(f"{partition_id} Executor (op={operation}): Retry {attempt + 1} - Generated new SQL")
                log_buffer.append(f"{partition_id} Reasoning: {merge_sql_decision.reasoning}")
                log_buffer.append(f"{partition_id} SQL: {merge_sql_decision.sql}")

    # All SQL attempts failed - return original rows
    log_buffer.append(f"{partition_id} ✗ Executor failed after {max_sql_attempts} attempts, returning original rows")
    return current_df, log_buffer


async def run_controller_executor_loop(
    pk_value: tuple,
    pk_df: pd.DataFrame,
    primary_key: list[str],
    question: str,
    schema: Tables,
    table_name: str,
    controller_chain,
    executor_chain,
    metadata: dict,
    model: str = "gpt-4.1",
    max_iterations: int = 5,
    max_controller_inspections: int = 3,
    max_executor_inspections: int = 3,
    max_sql_attempts: int = 3,
    verification_config: dict = None,
    context_generator_config: dict = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run the controller-executor loop for a single PK group.

    Loop:
    1. Controller inspects and chooses ONE operation
    2. If "stop": break
    3. Executor performs operation
    4. Repeat (up to max_iterations)

    Returns:
        (Final reconciled DataFrame for this PK group, list of operations performed, buffered log messages)
    """
    state = PKGroupState(pk_value, pk_df)
    log_buffer = []
    partition_id = f"[PK {pk_value}]"

    log_buffer.append(f"{partition_id} Processing: {len(pk_df)} rows, max {max_iterations} iterations")

    for iteration in range(max_iterations):
        # Controller decides next operation
        operation, log_buffer = await run_controller_for_pk(
            current_df=state.current_df,
            primary_key=primary_key,
            pk_value=pk_value,
            iteration=iteration,
            max_iterations=max_iterations,
            operations_history=state.operations_performed,
            question=question,
            schema=schema,
            table_name=table_name,
            controller_chain=controller_chain,
            metadata=metadata,
            max_controller_inspections=max_controller_inspections,
            log_buffer=log_buffer,
            partition_id=partition_id,
        )

        # Check for stop
        if operation == "stop":
            log_buffer.append(f"{partition_id} Controller chose 'stop' at iteration {iteration}")
            break

        # Record operation
        state.operations_performed.append(operation)

        # Executor performs operation
        updated_df, executor_logs = await run_executor_for_pk(
            current_df=state.current_df,
            operation=operation,
            primary_key=primary_key,
            pk_value=pk_value,
            question=question,
            schema=schema,
            table_name=table_name,
            executor_chain=executor_chain,
            metadata=metadata,
            model=model,
            max_executor_inspections=max_executor_inspections,
            max_sql_attempts=max_sql_attempts,
            verification_config=verification_config,
            context_generator_config=context_generator_config,
        )

        # Add executor logs to buffer
        log_buffer.extend(executor_logs)

        # Update state
        state.current_df = updated_df
        state.iteration = iteration + 1

        # Check if we reduced to 0 or 1 rows (natural stopping point)
        if len(updated_df) <= 1:
            log_buffer.append(f"{partition_id} Reduced to {len(updated_df)} rows, stopping")
            break

    log_buffer.append(
        f"{partition_id} Final result = {len(state.current_df)} rows after {len(state.operations_performed)} operations"
    )
    return state.current_df, state.operations_performed, log_buffer


async def process_single_pk_group(
    pk_value: tuple,
    pk_df: pd.DataFrame,
    primary_key: list[str],
    question: str,
    schema: Tables,
    table_name: str,
    controller_chain,
    executor_chain,
    metadata: dict,
    model: str = "gpt-4.1",
    max_iterations: int = 5,
    max_controller_inspections: int = 3,
    max_executor_inspections: int = 3,
    max_sql_attempts: int = 3,
    verification_config: dict = None,
    context_generator_config: dict = None,
) -> tuple[tuple, pd.DataFrame, list[str], bool, list[str]]:
    """
    Process a single PK group.

    If 1 row: pass through unchanged
    If >1 row: run controller-executor loop

    Returns:
        (pk_value, result_df, operations_performed, success, log_buffer)
    """
    try:
        if len(pk_df) == 1:
            # Single row - pass through
            partition_id = f"[PK {pk_value}]"
            log_buffer = [f"{partition_id} Single row, passing through"]
            return (pk_value, pk_df, [], True, log_buffer)

        # Multi-row group - run loop
        result_df, operations_performed, log_buffer = await run_controller_executor_loop(
            pk_value=pk_value,
            pk_df=pk_df,
            primary_key=primary_key,
            question=question,
            schema=schema,
            table_name=table_name,
            controller_chain=controller_chain,
            executor_chain=executor_chain,
            metadata=metadata,
            model=model,
            max_iterations=max_iterations,
            max_controller_inspections=max_controller_inspections,
            max_executor_inspections=max_executor_inspections,
            max_sql_attempts=max_sql_attempts,
            verification_config=verification_config,
            context_generator_config=context_generator_config,
        )

        return (pk_value, result_df, operations_performed, True, log_buffer)

    except Exception as e:
        partition_id = f"[PK {pk_value}]"
        error_log = [f"{partition_id} ✗ Processing failed: {e}"]
        # Return original rows on failure with reconciliation context indicating skip due to error
        pk_df_with_context = pk_df.copy()
        pk_df_with_context["__reconciliation_context__"] = (
            f'{{"operation": "skipped", "original_row_count": {len(pk_df)}, "final_row_count": {len(pk_df)}, "summary": "Reconciliation skipped due to code error, returning original rows", "information_discarded": "none - error occurred: {str(e)[:100]}"}}'
        )
        return (pk_value, pk_df_with_context, [], False, error_log)


# ============================================================================
# Main Entry Point
# ============================================================================


async def run_reconciliation(
    question: str,
    documents: list[Document],
    schema: Tables,
    table_data: pd.DataFrame,
    table_name: str,
    original_table_name: str,
    run_provenance: bool,
    metadata: dict,
    model_config: dict,
    reconciliation_config: dict = None,
) -> ExtractedTable:
    """
    Reconciliation Pipeline V2 - Main entry point.

    Pipeline:
    1. Select primary key for deduplication/consolidation
    2. Canonicalize primary key fields
    3. Split by PK groups and run parallel controller-executor loops

    Args:
        question: The question to answer
        documents: Source documents
        schema: Table schema
        table_data: Initial table data
        table_name: Name of the table in DuckDB
        original_table_name: Original table name
        run_provenance: Whether to track provenance (not implemented)
        metadata: Additional metadata
        model_config: Model configuration
        reconciliation_config: Configuration for reconciliation behavior

    Returns:
        ExtractedTable with merged data and __reconciliation_context__ field
        documenting all reconciliation operations performed
    """
    logger.info("=" * 80)
    logger.info("RECONCILIATION PIPELINE V2 START")
    logger.info("=" * 80)
    logger.info(f"Table: {table_name}")
    logger.info(f"Initial shape: {table_data.shape}")

    # Extract reconciliation config
    reconciliation_config = reconciliation_config or {}
    pk_selection_config = reconciliation_config.get("primary_key_selection", {})
    canonicalization_config = reconciliation_config.get("canonicalization", {})
    canonicalization_config.setdefault("document_level_max_cycles", 5)
    controller_executor_config = reconciliation_config.get("controller_executor_loop", {})
    verification_config = controller_executor_config.get("verification", {})
    context_generator_config = controller_executor_config.get("context_generator", {})
    non_pk_canon_config = reconciliation_config.get("non_pk_canonicalization", {})
    column_selector_config = non_pk_canon_config.get("column_selector", {})
    statistics_config = reconciliation_config.get("statistics", {})

    # Debug mode: save intermediate CSVs to a per-table subdirectory
    debug_mode = reconciliation_config.get("debug_mode", False)
    output_folder = Path(metadata.get("output_folder", ".")) if metadata else Path(".")
    if debug_mode:
        intermediate_dir = output_folder / "intermediate_tables" / original_table_name
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Get primary key selector version (default to v2)

    # Get initial table statistics
    initial_stats = get_table_stats(table_data, table_name)

    # Store reconciliation info in metadata
    if "reconciliation" not in metadata:
        metadata["reconciliation"] = {}

    reconciliation_metadata = metadata["reconciliation"]
    reconciliation_metadata["initial_stats"] = initial_stats

    if debug_mode:
        pre_recon_csv = intermediate_dir / "01_pre_reconciliation.csv"
        table_data.to_csv(pre_recon_csv, index=False)
        logger.info(f"[DEBUG] Saved pre-reconciliation table ({table_data.shape}): {pre_recon_csv}")

    # ========================================================================
    # PHASE 1: Primary Key Selection
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: PRIMARY KEY SELECTION")
    logger.info("=" * 80)

    # Check if primary key is provided in config
    provided_primary_key = pk_selection_config.get("primary_key", None)
    primary_key = None
    pk_info = None

    if provided_primary_key:
        # Validate that all provided primary key columns exist in the table
        table_columns = set(table_data.columns)
        provided_key_set = set(provided_primary_key)

        if provided_key_set.issubset(table_columns):
            # All columns exist - use provided primary key
            primary_key = provided_primary_key
            logger.info(f"✓ Using provided primary key from config: {primary_key}")
            logger.info("  Skipping primary key selection")

            pk_info = {
                "primary_key": primary_key,
                "reasoning": "Primary key provided in configuration",
                "method": "config_provided",
            }
            reconciliation_metadata["primary_key"] = {
                "fields": primary_key,
                "reasoning": pk_info["reasoning"],
                "method": "config_provided",
            }
        else:
            # Some columns don't exist - log warning and fall back to selection
            missing_cols = provided_key_set - table_columns
            logger.warning(f"Provided primary key columns not found in table: {missing_cols}")
            logger.warning(f"Available columns: {sorted(table_columns)}")
            logger.warning("Falling back to automatic primary key selection")
            provided_primary_key = None  # Clear to trigger selection below

    if primary_key is None:
        pk_selector = PrimaryKeySelector(model_config=model_config, pk_selection_config=pk_selection_config)

        with DuckSQLBasic() as duck_sql_conn:
            # Register table
            duck_sql_conn.register(
                table_data,
                table_name,
                schema=schema,
                schema_table_name=original_table_name,
            )

            # Create ExtractedTable wrapper for PK selector
            extracted_table = ExtractedTable(
                name=original_table_name,
                tables=schema,
                sql_query=None,
                dataframe=table_data,
                dataframe_table_name=table_name,
                table_str=str(table_data),
            )

            try:
                pk_selections = await pk_selector.select_primary_keys(
                    question=question,
                    tables=[extracted_table],
                    schema=schema,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                )

                if original_table_name in pk_selections:
                    pk_info = pk_selections[original_table_name]
                    primary_key = pk_info["primary_key"]
                    logger.info(f"✓ Selected Primary Key: {primary_key}")
                    logger.info(f"  Reasoning: {pk_info['reasoning']}")

                    # V1 returns 'query_count', V2 returns 'candidates_evaluated'
                    if "query_count" in pk_info:
                        logger.info(f"  Queries Used: {pk_info['query_count']}")
                        reconciliation_metadata["primary_key"] = {
                            "fields": primary_key,
                            "reasoning": pk_info["reasoning"],
                            "query_count": pk_info["query_count"],
                        }
                    else:
                        logger.info(f"  Candidates Evaluated: {pk_info.get('candidates_evaluated', 'N/A')}")
                        reconciliation_metadata["primary_key"] = {
                            "fields": primary_key,
                            "reasoning": pk_info["reasoning"],
                            "candidates_evaluated": pk_info.get("candidates_evaluated", 0),
                        }
                else:
                    logger.warning("No primary key selected, using all fields as fallback")
                    primary_key = [col for col in table_data.columns if not col.startswith("__")]
                    pk_info = {"primary_key": primary_key, "reasoning": "Fallback: all fields", "query_count": 0}
            except Exception as e:
                logger.error(f"Primary key selection failed: {e}")
                logger.warning("Using all fields as fallback primary key")
                primary_key = [col for col in table_data.columns if not col.startswith("__")]
                pk_info = {"primary_key": primary_key, "reasoning": f"Error fallback: {str(e)}", "query_count": 0}

    # ====================================================================
    # PHASE 2: Primary Key Canonicalization
    # ====================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: PRIMARY KEY CANONICALIZATION")
    logger.info("=" * 80)
    logger.info(f"Canonicalizing {len(primary_key)} primary key field(s)")

    canonicalizer = FieldCanonicalizer(model_config=model_config, canonicalization_config=canonicalization_config)
    canonicalized_table = table_data
    canon_info = None

    with DuckSQLBasic() as duck_sql_conn:
        # Register table
        duck_sql_conn.register(
            table_data,
            table_name,
            schema=schema,
            schema_table_name=original_table_name,
        )

        # Create ExtractedTable wrapper
        extracted_table = ExtractedTable(
            name=original_table_name,
            tables=schema,
            sql_query=None,
            dataframe=table_data,
            dataframe_table_name=table_name,
            table_str=str(table_data),
        )

        try:
            # Check canonicalization mode config
            canonicalization_mode = canonicalization_config.get("mode", "two_pass")
            inspections_per_field = canonicalization_config.get("inspections_per_field", 5)

            if canonicalization_mode == "two_pass":
                logger.info("Using two-pass canonicalization (document-level → global)")
                canonicalized_table, canon_info = await canonicalizer.canonicalize_table_two_pass(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                    inspections_per_field=inspections_per_field,
                )
            elif canonicalization_mode == "document_only":
                logger.info("Using document-level canonicalization only")
                canonicalized_table, canon_info = await canonicalizer.canonicalize_table_by_document(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                    inspections_per_field=inspections_per_field,
                )
            else:  # "global_only" or default
                logger.info("Using global canonicalization only")
                canonicalized_table, canon_info = await canonicalizer.canonicalize_table(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                    inspections_per_field=inspections_per_field,
                )

            # Log summary based on mode
            if canonicalization_mode == "two_pass":
                pass1_info = canon_info.get("pass_1_document_level", {})
                pass2_info = canon_info.get("pass_2_global_level", {})
                docs_processed = pass1_info.get("documents_processed", 0)
                pass2_count = pass2_info.get("canonicalization_count", 0)
                logger.info("✓ Two-pass canonicalization complete")
                logger.info(f"  Pass 1: {docs_processed} documents processed")
                logger.info(f"  Pass 2: {pass2_count} global canonicalization(s)")
            elif canonicalization_mode == "document_only":
                docs_processed = canon_info.get("documents_processed", 0)
                logger.info(f"✓ Document-level canonicalization complete: {docs_processed} documents processed")
            else:
                logger.info(f"✓ Canonicalized {canon_info.get('canonicalization_count', 0)} field(s)")
                if canon_info.get("canonicalized_fields"):
                    logger.info(f"  Fields: {canon_info['canonicalized_fields']}")
                if canon_info.get("skipped_fields"):
                    logger.info(f"  Skipped: {canon_info['skipped_fields']}")

            # Debug mode: save post-canonicalization CSV
            if debug_mode:
                canon_csv = intermediate_dir / "02_post_pk_canonicalization.csv"
                canonicalized_table.to_csv(canon_csv, index=False)
                logger.info(f"[DEBUG] Saved post-PK-canonicalization table ({canonicalized_table.shape}): {canon_csv}")

            reconciliation_metadata["pk_canonicalization"] = canon_info

            # Update extracted table with canonicalized data
            extracted_table.dataframe = canonicalized_table

        except Exception as e:
            logger.error(f"Primary key canonicalization failed: {e}")
            logger.warning("Continuing with non-canonicalized primary key fields")
            reconciliation_metadata["pk_canonicalization"] = {"error": str(e)}

        # ====================================================================
        # PHASE 2.5: Handle NULL Primary Keys
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2.5: NULL PRIMARY KEY HANDLING")
        logger.info("=" * 80)
        logger.info(f"Handling NULL values in {len(primary_key)} primary key field(s)")

        null_handler = NullPKHandler(
            model_config=model_config, null_pk_config=canonicalization_config.get("null_handling", {})
        )

        try:
            # Re-register table with canonicalized data
            duck_sql_conn.register(canonicalized_table, table_name)

            # Update extracted table with canonicalized data before null handling
            extracted_table.dataframe = canonicalized_table

            # Handle NULL PKs based on canonicalization mode
            if canonicalization_mode == "two_pass":
                logger.info("Using two-pass NULL handling (document-level → global)")
                canonicalized_table, null_handling_info = await null_handler.handle_null_primary_keys_two_pass(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                )
            elif canonicalization_mode == "document_only":
                logger.info("Using document-level NULL handling only")
                canonicalized_table, null_handling_info = await null_handler.handle_null_primary_keys_by_document(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    metadata=metadata,
                )
            else:  # "global_only" or default
                logger.info("Using global NULL handling only")
                canonicalized_table, null_handling_info = await null_handler.handle_null_primary_keys(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    duck_sql_conn=duck_sql_conn,
                    metadata=metadata,
                )

            # Log summary based on mode
            if canonicalization_mode == "two_pass":
                pass1_info = null_handling_info.get("pass_1_document_level", {})
                pass2_info = null_handling_info.get("pass_2_global_level", {})
                docs_processed = pass1_info.get("documents_processed", 0)
                pass2_nulls = pass2_info.get("total_null_rows", 0)
                logger.info("✓ Two-pass NULL handling complete")
                logger.info(f"  Pass 1: {docs_processed} documents processed")
                logger.info(f"  Pass 2: {pass2_nulls} global NULL rows processed")
            elif canonicalization_mode == "document_only":
                docs_processed = null_handling_info.get("documents_processed", 0)
                logger.info(f"✓ Document-level NULL handling complete: {docs_processed} documents processed")
            else:
                logger.info(f"✓ Processed {null_handling_info['total_null_rows']} NULL row(s)")

            logger.info(f"  Filled: {null_handling_info.get('filled_count', 0)}")
            logger.info(f"  Discarded: {null_handling_info.get('discarded_count', 0)}")
            logger.info(f"  Placeholders: {null_handling_info.get('placeholder_count', 0)}")

            if debug_mode:
                null_csv = intermediate_dir / "03a_post_null_pk_handling.csv"
                canonicalized_table.to_csv(null_csv, index=False)
                logger.info(f"[DEBUG] Saved post-NULL-PK-handling table ({canonicalized_table.shape}): {null_csv}")

            reconciliation_metadata["null_pk_handling"] = null_handling_info

            # Update extracted table with NULL-handled data
            extracted_table.dataframe = canonicalized_table

        except Exception as e:
            logger.error(f"NULL PK handling failed: {e}")
            logger.warning("Continuing with NULL values in primary key")
            reconciliation_metadata["null_pk_handling"] = {"error": str(e)}

        # ====================================================================
        # PHASE 2.6: Handle NULL Non-Primary-Key Columns (Document Level)
        # ====================================================================
        if canonicalization_mode in ["two_pass", "document_only"]:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2.6: DOCUMENT-LEVEL NON-PK NULL HANDLING")
            logger.info("=" * 80)
            logger.info("Filling gaps in non-primary-key data columns within each document")

            try:
                # Re-register table with PK-null-handled data
                duck_sql_conn.register(canonicalized_table, table_name)
                extracted_table.dataframe = canonicalized_table

                canonicalized_table, non_pk_null_info = await null_handler.handle_null_non_pk_columns_by_document(
                    table=extracted_table,
                    schema=schema,
                    primary_key=primary_key,
                    metadata=metadata,
                )

                logger.info(f"✓ Processed non-PK NULLs in {non_pk_null_info.get('documents_processed', 0)} documents")
                logger.info(f"  Rows with filled NULLs: {non_pk_null_info.get('rows_modified', 0)}")

                if debug_mode:
                    non_pk_null_csv = intermediate_dir / "03b_post_non_pk_null_handling.csv"
                    canonicalized_table.to_csv(non_pk_null_csv, index=False)
                    logger.info(
                        f"[DEBUG] Saved post-non-PK-null-handling table ({canonicalized_table.shape}): {non_pk_null_csv}"
                    )

                reconciliation_metadata["non_pk_null_handling"] = non_pk_null_info

                # Update extracted table with non-PK-null-handled data
                extracted_table.dataframe = canonicalized_table

            except Exception as e:
                logger.error(f"Non-PK NULL handling failed: {e}")
                logger.warning("Continuing with NULLs in non-PK columns")
                reconciliation_metadata["non_pk_null_handling"] = {"error": str(e)}
        else:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2.6: DOCUMENT-LEVEL NON-PK NULL HANDLING")
            logger.info("=" * 80)
            logger.info("Non-PK NULL handling skipped (global_only mode)")
            reconciliation_metadata["non_pk_null_handling"] = {"skipped": True, "reason": "global_only mode"}

    # ========================================================================
    # PHASE 3: Per-PK Group Processing
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: PER-PK GROUP RECONCILIATION")
    logger.info("=" * 80)

    # Initialize LLM chains - use merge_tables model for both controller and executor
    model = model_config.get("merge_tables", {}).get("model", "gpt-4.1")
    llm_client = get_llm_client(model=model, temperature=0.0)

    # Controller chain
    controller_template = load_fewshot_prompt_template(
        template_file="sliders/reconcilation/controller_v2.prompt",
        template_blocks=[],
    )
    controller_chain = controller_template | llm_client.with_structured_output(ControllerDecisionV2)

    # Executor chain
    executor_template = load_fewshot_prompt_template(
        template_file="sliders/reconcilation/executor_v2.prompt",
        template_blocks=[],
    )
    executor_chain = executor_template | llm_client.with_structured_output(ExecutorDecisionV2)

    # Add number_instances column before splitting into PK groups
    canonicalized_table["number_instances"] = 1
    logger.info("Added number_instances column (initialized to 1 for all rows)")

    # Split table by PK groups
    logger.info(f"Splitting table by primary key: {primary_key}")
    pk_groups = split_by_pk_groups(canonicalized_table, primary_key)

    # Get placeholder text from null_handling config (default to 'UNKNOWN')
    null_handling_config = canonicalization_config.get("null_handling", {})
    placeholder_text = null_handling_config.get("placeholder_text", "UNKNOWN")

    # Separate groups with placeholder PKs - these should pass through without reconciliation
    placeholder_pk_groups = {}
    valid_pk_groups = {}

    for pk_value, df in pk_groups.items():
        # Check if any field in the PK tuple contains the placeholder text
        pk_tuple = pk_value if isinstance(pk_value, tuple) else (pk_value,)
        has_placeholder = any(str(pk_field) == placeholder_text for pk_field in pk_tuple)

        if has_placeholder:
            placeholder_pk_groups[pk_value] = df
        else:
            valid_pk_groups[pk_value] = df

    single_row_groups = {pk: df for pk, df in valid_pk_groups.items() if len(df) == 1}
    multi_row_groups = {pk: df for pk, df in valid_pk_groups.items() if len(df) > 1}

    logger.info(f"Total PK groups: {len(pk_groups)}")
    if placeholder_pk_groups:
        logger.info(
            f"  Groups with placeholder PK '{placeholder_text}': {len(placeholder_pk_groups)} (pass through, {sum(len(df) for df in placeholder_pk_groups.values())} rows)"
        )
    logger.info(f"  Single-row groups: {len(single_row_groups)} (pass through)")
    logger.info(f"  Multi-row groups: {len(multi_row_groups)} (need reconciliation)")

    # Process multi-row groups in parallel
    if multi_row_groups:
        logger.info(f"\nProcessing {len(multi_row_groups)} multi-row PK groups in parallel...")

        # Extract controller-executor loop config
        max_iterations = controller_executor_config.get("max_iterations", 5)
        max_controller_inspections = controller_executor_config.get("max_controller_inspections", 3)
        max_executor_inspections = controller_executor_config.get("max_executor_inspections", 3)
        max_sql_attempts = controller_executor_config.get("max_sql_attempts", 3)

        tasks = [
            process_single_pk_group(
                pk_value=pk_value,
                pk_df=pk_df,
                primary_key=primary_key,
                question=question,
                schema=schema,
                table_name=table_name,
                controller_chain=controller_chain,
                executor_chain=executor_chain,
                metadata=metadata,
                model=model,
                max_iterations=max_iterations,
                max_controller_inspections=max_controller_inspections,
                max_executor_inspections=max_executor_inspections,
                max_sql_attempts=max_sql_attempts,
                verification_config=verification_config,
                context_generator_config=context_generator_config,
            )
            for pk_value, pk_df in multi_row_groups.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results and print buffered logs sequentially
        processed_groups = {}
        failed_groups = []
        all_operations: list[str] = []
        multi_row_group_initial_rows = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"PK group processing exception: {result}")
                failed_groups.append({"error": str(result)})
            else:
                pk_value, result_df, operations, success, log_buffer = result

                # Print buffered logs for this PK group with prominent header
                logger.info("")  # Blank line for separation
                logger.info("=" * 80)
                # First log message is the partition header, make it stand out
                if log_buffer:
                    logger.info(log_buffer[0])
                    logger.info("=" * 80)
                    # Print rest of logs
                    for log_msg in log_buffer[1:]:
                        logger.info(log_msg)
                    logger.info("=" * 80)
                    logger.info("")  # Blank line after partition

                if success:
                    processed_groups[pk_value] = result_df
                    all_operations.extend(operations)
                    # Count initial rows for this multi-row group
                    if pk_value in multi_row_groups:
                        multi_row_group_initial_rows += len(multi_row_groups[pk_value])
                else:
                    # For failed groups, use the original data from multi_row_groups and add error context
                    if pk_value in multi_row_groups:
                        original_df = multi_row_groups[pk_value].copy()
                        original_df["__reconciliation_context__"] = (
                            f'{{"operation": "skipped", "original_row_count": {len(original_df)}, "final_row_count": {len(original_df)}, "summary": "Reconciliation skipped due to code error, returning original rows", "information_discarded": "none"}}'
                        )
                        processed_groups[pk_value] = original_df
                        multi_row_group_initial_rows += len(multi_row_groups[pk_value])
                        logger.info(f"[PK {pk_value}] Failed - using original {len(original_df)} rows")
                    failed_groups.append({"pk_value": pk_value})

        logger.info("\nProcessing complete:")
        logger.info(f"  Succeeded: {len(processed_groups)}/{len(multi_row_groups)}")
        logger.info(f"  Failed: {len(failed_groups)}")
    else:
        processed_groups = {}
        failed_groups = []
        all_operations: list[str] = []
        multi_row_group_initial_rows = 0

    # Concatenate all results
    all_dfs = []

    # Add placeholder PK groups first (pass through without reconciliation)
    for pk_value, df in placeholder_pk_groups.items():
        # Add __reconciliation_context__ field for placeholder PK groups
        df_copy = df.copy()
        df_copy["__reconciliation_context__"] = (
            f'{{"operation": "skipped", "original_row_count": {len(df)}, "final_row_count": {len(df)}, "summary": "Skipped - primary key contains placeholder value ({placeholder_text})", "information_discarded": "none"}}'
        )
        all_dfs.append(df_copy)

    # Add single-row groups (add reconciliation context indicating no reconciliation needed)
    for pk_value, df in single_row_groups.items():
        # Add __reconciliation_context__ field for single-row groups
        df_copy = df.copy()
        df_copy["__reconciliation_context__"] = (
            '{"operation": "none", "original_row_count": 1, "final_row_count": 1, "summary": "Single row, no reconciliation needed", "information_discarded": "none"}'
        )
        all_dfs.append(df_copy)

    # Add processed multi-row groups (these already have __reconciliation_context__ from executor)
    for pk_value, df in processed_groups.items():
        if len(df) > 0:  # Skip discarded groups
            all_dfs.append(df)

    if all_dfs:
        final_table = pd.concat(all_dfs, ignore_index=True)
    else:
        # Create empty dataframe with __reconciliation_context__ column
        columns = list(canonicalized_table.columns) + ["__reconciliation_context__"]
        final_table = pd.DataFrame(columns=columns)

    logger.info(f"\nFinal reconciled table: {len(final_table)} rows")
    logger.info(f"Row reduction: {len(table_data)} → {len(final_table)}")

    if debug_mode:
        controller_csv = intermediate_dir / "04_post_controller_executor.csv"
        final_table.to_csv(controller_csv, index=False)
        logger.info(f"[DEBUG] Saved post-controller-executor table ({final_table.shape}): {controller_csv}")

    # Verify reconciliation context field exists
    if "__reconciliation_context__" in final_table.columns:
        logger.info("✓ Reconciliation context field added to all rows")
        # Count rows with different operations
        if len(final_table) > 0:
            try:
                import json

                operation_types = {}
                for _, row in final_table.iterrows():
                    try:
                        ctx = json.loads(row["__reconciliation_context__"])
                        op = ctx.get("operation", "unknown")
                        operation_types[op] = operation_types.get(op, 0) + 1
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                if operation_types:
                    logger.info(f"  Operations breakdown: {operation_types}")
            except Exception as e:
                logger.debug(f"Could not parse reconciliation context: {e}")
    else:
        logger.warning("⚠️ Reconciliation context field missing from final table")

    # Store metadata
    reconciliation_metadata["pk_groups"] = {
        "total": len(pk_groups),
        "single_row": len(single_row_groups),
        "multi_row": len(multi_row_groups),
        "processed": len(processed_groups),
        "failed": len(failed_groups),
    }
    reconciliation_metadata["multi_row_group_initial_rows"] = multi_row_group_initial_rows
    reconciliation_metadata["all_operations"] = deepcopy(all_operations)
    reconciliation_metadata["final_row_count"] = len(final_table)

    # ========================================================================
    # PHASE 4: Non-Primary-Key Column Canonicalization
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: NON-PRIMARY-KEY COLUMN CANONICALIZATION")
    logger.info("=" * 80)

    non_pk_canon_config = reconciliation_config.get("non_pk_canonicalization", {})
    enable_non_pk_canon = non_pk_canon_config.get("enable", False)

    if enable_non_pk_canon and len(final_table) > 0:
        logger.info("Non-PK column canonicalization enabled")

        try:
            # Step 1: Select columns to canonicalize
            columns_to_canonicalize = await select_columns_for_canonicalization(
                table_df=final_table,
                primary_key=primary_key,
                table_name=table_name,
                schema=schema,
                metadata=metadata,
                model_config=model_config,
                column_selector_config=column_selector_config,
            )

            logger.info(f"Identified {len(columns_to_canonicalize)} columns for canonicalization")

            if columns_to_canonicalize:
                # Step 2: Canonicalize each column iteratively
                logger.info(f"Canonicalizing columns: {columns_to_canonicalize}")

                with DuckSQLBasic() as duck_sql_conn:
                    duck_sql_conn.register(final_table, table_name)

                    extracted_table = ExtractedTable(
                        name=original_table_name,
                        tables=schema,
                        sql_query=None,
                        dataframe=final_table,
                        dataframe_table_name=table_name,
                        table_str=str(final_table),
                    )

                    canonicalizer = FieldCanonicalizer(
                        model_config=model_config, canonicalization_config=non_pk_canon_config
                    )

                    inspections_per_field = non_pk_canon_config.get("inspections_per_field", 5)
                    final_table, non_pk_canon_info = await canonicalizer.canonicalize_table(
                        table=extracted_table,
                        schema=schema,
                        primary_key=columns_to_canonicalize,  # Reuse parameter name but pass our column list
                        duck_sql_conn=duck_sql_conn,
                        metadata=metadata,
                        inspections_per_field=inspections_per_field,
                    )

                    logger.info(f"✓ Canonicalized {non_pk_canon_info['canonicalization_count']} non-PK column(s)")
                    if non_pk_canon_info.get("canonicalized_fields"):
                        logger.info(f"  Canonicalized: {non_pk_canon_info['canonicalized_fields']}")
                    if non_pk_canon_info.get("skipped_fields"):
                        logger.info(f"  Skipped: {non_pk_canon_info['skipped_fields']}")

                    if debug_mode:
                        non_pk_canon_csv = intermediate_dir / "05a_post_non_pk_canonicalization.csv"
                        final_table.to_csv(non_pk_canon_csv, index=False)
                        logger.info(
                            f"[DEBUG] Saved post-non-PK-canonicalization table ({final_table.shape}): {non_pk_canon_csv}"
                        )

                    reconciliation_metadata["non_pk_canonicalization"] = non_pk_canon_info
            else:
                logger.info("No columns identified for canonicalization")
                reconciliation_metadata["non_pk_canonicalization"] = {
                    "canonicalized_fields": [],
                    "skipped_fields": [],
                    "canonicalization_count": 0,
                }
        except Exception as e:
            logger.error(f"Non-PK column canonicalization failed: {e}")
            logger.warning("Continuing with non-canonicalized columns")
            reconciliation_metadata["non_pk_canonicalization"] = {"error": str(e)}
    else:
        if not enable_non_pk_canon:
            logger.info("Non-PK column canonicalization disabled")
        else:
            logger.info("Non-PK column canonicalization skipped (empty table)")
        reconciliation_metadata["non_pk_canonicalization"] = {"enabled": False}

    # ========================================================================
    # Build and Save Statistics
    # ========================================================================

    # Count operations by type
    all_operations_list = reconciliation_metadata.get("all_operations", [])
    operation_counts = {
        "deduplicate": all_operations_list.count("deduplicate"),
        "aggregate": all_operations_list.count("aggregate"),
        "resolve_conflicts": all_operations_list.count("resolve_conflicts"),
        "canonicalize": all_operations_list.count("canonicalize"),
    }

    # Build canonicalization stats per field
    canonicalization_stats = {}
    if canon_info and "field_canonicalizations" in canon_info:
        for field_name, field_info in canon_info["field_canonicalizations"].items():
            canonicalization_stats[field_name] = {
                "total_operations": field_info.get("total_queries", 0),
                "canonicalized": field_info.get("canonicalized", False),
            }

    # Build non-PK canonicalization stats
    non_pk_canonicalization_stats = {}
    non_pk_canon_info_data = reconciliation_metadata.get("non_pk_canonicalization", {})
    if "field_canonicalizations" in non_pk_canon_info_data:
        for field_name, field_info in non_pk_canon_info_data["field_canonicalizations"].items():
            non_pk_canonicalization_stats[field_name] = {
                "total_operations": field_info.get("total_queries", 0),
                "canonicalized": field_info.get("canonicalized", False),
            }

    # Calculate total rows for multi-row groups
    multi_row_group_final_rows = sum(len(df) for df in processed_groups.values() if len(df) > 0)

    # Build comprehensive statistics
    stats = {
        original_table_name: {
            "primary_key": primary_key,
            "canonicalization": {
                "stats_per_field": canonicalization_stats,
                "total_fields_canonicalized": len([f for f in canonicalization_stats.values() if f["canonicalized"]]),
                "total_fields_processed": len(canonicalization_stats),
            },
            "non_pk_canonicalization": {
                "enabled": enable_non_pk_canon,
                "stats_per_field": non_pk_canonicalization_stats,
                "total_fields_canonicalized": len(
                    [f for f in non_pk_canonicalization_stats.values() if f["canonicalized"]]
                ),
                "total_fields_processed": len(non_pk_canonicalization_stats),
                "columns_identified": non_pk_canon_info_data.get("canonicalized_fields", [])
                + non_pk_canon_info_data.get("skipped_fields", []),
            },
            "row_grouping": {
                "unique_primary_key_values": len(pk_groups),
                "single_row_groups": len(single_row_groups),
                "multi_row_groups": len(multi_row_groups),
            },
            "multi_row_processing": {
                "initial_total_rows": multi_row_group_initial_rows,
                "final_total_rows": multi_row_group_final_rows,
                "rows_reduced": multi_row_group_initial_rows - multi_row_group_final_rows,
            },
            "controller_actions": operation_counts,
            "final_stats": {
                "initial_total_rows": len(table_data),
                "final_total_rows": len(final_table),
                "total_rows_reduced": len(table_data) - len(final_table),
            },
        }
    }

    # Save statistics to JSON file
    enable_statistics = statistics_config.get("enable", True)
    statistics_filename = statistics_config.get("filename", "reconciliation_stats.json")

    if enable_statistics and metadata and "output_folder" in metadata:
        output_folder = Path(metadata["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        stats_file = output_folder / statistics_filename

        # Load existing stats if file exists, to support multiple tables
        existing_stats = {}
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    existing_stats = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing stats file: {e}")

        # Merge stats
        existing_stats.update(stats)

        # Write updated stats
        with open(stats_file, "w") as f:
            json.dump(existing_stats, f, indent=2)

        logger.info(f"Statistics saved to: {stats_file}")

    logger.info("=" * 80)
    logger.info("RECONCILIATION PIPELINE V2 COMPLETE")
    logger.info("=" * 80)

    if debug_mode:
        final_csv = intermediate_dir / "05_final_table.csv"
        final_table.to_csv(final_csv, index=False)
        logger.info(f"[DEBUG] Saved final reconciled table ({final_table.shape}): {final_csv}")
        logger.info(f"[DEBUG] All intermediate tables saved to: {intermediate_dir}")

    # Store stats in metadata for return
    reconciliation_metadata["stats"] = stats[original_table_name]

    # Return as ExtractedTable
    return ExtractedTable(
        name=original_table_name,
        tables=schema,
        sql_query=None,
        dataframe=final_table,
        dataframe_table_name=table_name,
        table_str=str(final_table),
    )
