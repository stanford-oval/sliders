"""SQL-first inspect answer strategy for scalable question answering."""

from typing import Literal, Optional
from pydantic import BaseModel

from sliders.llm_models import Tables
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.llm_tools.sql import run_sql_query
from sliders.utils import (
    get_table_stats,
    format_table_stats,
    format_sql_result,
    prepare_schema_for_template,
    prepare_table_stats_for_template,
)


def get_query_strategy(row_count: int) -> dict:
    """
    Determine query strategy based on table size.

    Returns dict with:
        - first_query_limit: Suggested row limit for initial sample query
        - subsequent_limit: Row limit for follow-up queries
        - max_queries: Total query budget
        - strategy: Strategy name
    """
    if row_count <= 100:
        # Small tables: Show everything
        return {
            "first_query_limit": row_count,
            "subsequent_limit": row_count,
            "max_queries": 10,
            "strategy": "full_table",
        }
    else:
        return {"first_query_limit": 100, "subsequent_limit": 50, "max_queries": 10, "strategy": "sql_focused"}


class QueryDecision(BaseModel):
    """Decision from the query generator: run another query or finalize."""

    reasoning: str
    action: Literal["query", "finalize"]
    sql: Optional[str] = None  # SQL query if action is "query"


class CitationSQL(BaseModel):
    """SQL query for selecting rows with provenance/citation information."""

    reasoning: str
    action: Literal["execute", "finalize"]  # execute = try SQL and inspect, finalize = accept current result
    sql: str  # SQL query to select rows with provenance columns (required if action=execute)


async def run_inspect_answer(
    question: str,
    tables: list,
    schema: Tables,
    duck_sql_conn,
    metadata: dict,
    model_config: dict,
    tool_output_chain,
    reconciliation_stats: dict = None,
    inspect_answer_config: dict = None,
) -> str:
    """
    Answer question using SQL-first inspect approach.

    Args:
        question: The question to answer
        tables: List of Table objects with dataframes
        schema: Schema information
        duck_sql_conn: DuckSQLBasic connection with registered tables
        metadata: Metadata dictionary for logging
        model_config: Model configuration
        tool_output_chain: Chain for verbalizing SQL results (not used, we create our own)
        reconciliation_stats: Statistics from reconciliation pipeline (optional)
        inspect_answer_config: Configuration for inspect_answer features (optional)
            - enable_citation_generation: bool (default: False)
            - enable_reconciliation_stats_verbalization: bool (default: False)

    Returns:
        Final answer string
    """
    answer_start_time = __import__("time").time()

    # Extract configuration
    inspect_answer_config = inspect_answer_config or {}
    enable_citation = inspect_answer_config.get("enable_citation_generation", False)
    enable_recon_stats = inspect_answer_config.get("enable_reconciliation_stats_verbalization", False)

    # Get table statistics and determine query strategy
    table_stats_list = []
    max_row_count = 0
    for table in tables:
        if table.dataframe is not None:
            stats = get_table_stats(table.dataframe, table.dataframe_table_name)
            table_stats_list.append({"table_name": table.dataframe_table_name, "stats": format_table_stats(stats)})
            max_row_count = max(max_row_count, stats["row_count"])

    # Determine query strategy based on largest table
    strategy = get_query_strategy(max_row_count)
    logger.info("=== SQL Inspect Answer ===")
    logger.info(f"Question: {question[:100]}...")
    logger.info(f"Tables: {[t.dataframe_table_name for t in tables if t.dataframe is not None]}")
    logger.info(f"Max table size: {max_row_count} rows")
    logger.info(f"Strategy: {strategy['strategy'].upper()}")
    logger.info(f"  - First query limit: {strategy['first_query_limit']} rows")
    logger.info(f"  - Subsequent limit: {strategy['subsequent_limit']} rows")
    logger.info(f"  - Query budget: {strategy['max_queries']}")

    # Initialize LLM chains
    # Prefer inspect_answer-specific model override, then fall back to main answer model.
    inspect_model_cfg = dict(inspect_answer_config.get("model_config", {}))
    if not inspect_model_cfg:
        inspect_model_cfg = dict(model_config.get("answer", {}))
    if "model" not in inspect_model_cfg:
        inspect_model_cfg["model"] = "gpt-4.1"
    logger.info(
        "Inspect answer model config: "
        f"model={inspect_model_cfg.get('model')} "
        f"temperature={inspect_model_cfg.get('temperature', 'default')} "
        f"max_tokens={inspect_model_cfg.get('max_tokens', 'default')}"
    )
    llm_client = get_llm_client(**inspect_model_cfg)

    query_template = load_fewshot_prompt_template(
        template_file="sliders/inspect_answer/query_generator.prompt",
        template_blocks=[],
    )
    query_chain = query_template | llm_client.with_structured_output(QueryDecision)

    # Query loop
    max_queries = strategy["max_queries"]
    max_retries = 3
    query_history = []  # List of (reasoning, sql, result) tuples

    for query_num in range(max_queries):
        queries_remaining = max_queries - query_num
        is_first_query = query_num == 0
        logger.info(f"--- Query {query_num + 1}/{max_queries} ({queries_remaining} remaining) ---")

        # Determine row limit for this query
        current_row_limit = strategy["first_query_limit"] if is_first_query else strategy["subsequent_limit"]
        logger.info(f"Row limit for this query: {current_row_limit}")

        # Try to generate and execute a query with retries
        query_attempts = []
        sql_error_feedback = None
        query_successful = False

        for retry in range(max_retries):
            if retry > 0:
                logger.info(f"  Retry {retry + 1}/{max_retries}")

            handler = LoggingHandler(
                prompt_file="sliders/inspect_answer/query_generator.prompt",
                metadata={
                    "question": question,
                    "stage": f"query_{query_num + 1}_attempt_{retry + 1}",
                    "question_id": metadata.get("question_id", None),
                    **(metadata or {}),
                },
            )

            # Format query history
            formatted_history = []
            for idx, (reasoning, sql, result) in enumerate(query_history):
                formatted_history.append(f"Query {idx + 1}:\nReasoning: {reasoning}\nSQL: {sql}\nResult:\n{result}\n")

            # Prepare error feedback if this is a retry
            if query_attempts:
                feedback_parts = []
                for idx, (sql, err) in enumerate(query_attempts):
                    feedback_parts.append(f"Attempt {idx + 1} failed:\nSQL: {sql}\nError: {err}\n")
                sql_error_feedback = (
                    "\n".join(feedback_parts) + f"\nThis is attempt {retry + 1}/{max_retries}. Please fix the SQL."
                )

            # Generate query decision
            decision = await query_chain.ainvoke(
                {
                    "question": question,
                    "schema": prepare_schema_for_template(schema),
                    "table_stats": prepare_table_stats_for_template(table_stats_list),
                    "query_history": "\n".join(formatted_history) if formatted_history else "No queries executed yet",
                    "queries_remaining": queries_remaining,
                    "sql_error_feedback": sql_error_feedback or "No previous SQL errors",
                    "strategy": strategy["strategy"],
                    "is_first_query": is_first_query,
                    "row_limit": current_row_limit,
                    # Citation mode variables (not used in normal query generation but required by template)
                    "citation_mode": False,
                    "final_answer": "",
                    "finalization_reasoning": "",
                    "citation_attempts_history": "",
                    "attempts_remaining": 0,
                },
                config={"callbacks": [handler]},
            )

            if decision.action == "finalize":
                logger.info(f"✓ Agent finalizing after {len(query_history)} successful queries")
                logger.info(f"  Reasoning: {decision.reasoning[:120]}...")
                break

            # Execute query
            if decision.sql:
                logger.info(f"Running SQL: {decision.sql[:100]}{'...' if len(decision.sql) > 100 else ''}")
                result, error = run_sql_query(
                    decision.sql, duck_sql_conn, row_limit=current_row_limit, output_format="formatted"
                )

                if not error:
                    # Success - add to history (reasoning, sql, result)
                    query_history.append((decision.reasoning, decision.sql, result))
                    # Log result info
                    if "No results" in result:
                        logger.info("  → No results returned")
                    elif "First" in result:
                        logger.info(f"  → Returned truncated results, showing first {current_row_limit}")
                    else:
                        logger.info("  → Returned all results")
                    logger.info("✓ Query successful")
                    query_successful = True
                    break
                else:
                    # Error - add to attempts and retry
                    logger.warning(f"  → SQL Error: {result}")
                    query_attempts.append((decision.sql, result))

                    if retry == max_retries - 1:
                        logger.error(f"✗ All {max_retries} attempts failed")
                        # Continue to next query or finalize
            else:
                logger.warning("Agent chose to query but provided no SQL")
                break

        # After retry loop, check if we should finalize
        if decision.action == "finalize":
            break
        if not query_successful:
            logger.warning(f"Query {query_num + 1} failed, moving to next query")
            break

    # Verbalize results
    logger.info("=== Verbalization ===")
    if query_history:
        logger.info(f"Verbalizing {len(query_history)} successful query results")

        # Format as interleaved reasoning + query + result
        formatted_steps = []
        for i, (reasoning, sql, result) in enumerate(query_history):
            step = f"## Step {i + 1}\n\n**Reasoning:** {reasoning}\n\n**SQL Query:**\n```sql\n{sql}\n```\n\n**Result:**\n```\n{result}\n```"
            formatted_steps.append(step)

        tool_call = "\n\n".join(formatted_steps)
        tool_output = "See interleaved results above"

        from langchain_core.output_parsers import StrOutputParser

        verbalize_template = load_fewshot_prompt_template(
            template_file="sliders/inspect_answer/verbalize_results.prompt",
            template_blocks=[],
        )
        verbalize_chain = verbalize_template | llm_client | StrOutputParser()

        verbalize_handler = LoggingHandler(
            prompt_file="sliders/inspect_answer/verbalize_results.prompt",
            metadata={
                "question": question,
                "tool_call": tool_call,
                "tool_output": tool_output,
                "question_id": metadata.get("question_id", None),
                "stage": "verbalize_inspect_answer",
            },
        )

        final_answer = await verbalize_chain.ainvoke(
            {
                "question": question,
                "tool_call": tool_call,
                "tool_output": tool_output,
                "classes": prepare_schema_for_template(schema),
                # Citation mode variables (not used in normal verbalization but required by template)
                "citation_mode": False,
                "generated_answer": "",
                "citation_sql": "",
                "citation_data": "",
            },
            config={"callbacks": [verbalize_handler]},
        )
    else:
        logger.warning("No successful queries executed, providing error message")
        final_answer = "Unable to answer the question due to SQL query failures."

    # ========================================================================
    # CITATION SQL GENERATION
    # ========================================================================
    logger.info("=== Citation SQL Generation ===")
    if enable_citation:
        logger.info("Citation generation enabled")
    else:
        logger.info("Citation generation disabled, skipping")

    if enable_citation and query_history:
        # Format query history for citation stage
        formatted_steps = []
        for i, (reasoning, sql, result) in enumerate(query_history):
            step = f"## Step {i + 1}\n\n**Reasoning:** {reasoning}\n\n**SQL Query:**\n```sql\n{sql}\n```\n\n**Result:**\n```\n{result}\n```"
            formatted_steps.append(step)

        citation_query_history = "\n\n".join(formatted_steps)

        # Get finalization reasoning from last decision
        finalization_reasoning = decision.reasoning if decision else "Agent finalized investigation."

        # Create citation SQL chain
        citation_template = load_fewshot_prompt_template(
            template_file="sliders/inspect_answer/query_generator.prompt",
            template_blocks=[],
        )
        citation_chain = citation_template | llm_client.with_structured_output(CitationSQL)

        # Citation SQL generation with retry loop (up to 3 attempts)
        max_citation_attempts = 3
        citation_attempts = []  # Track (sql, result) pairs
        citation_result = None
        final_citation_sql = None
        final_citation_reasoning = None

        try:
            for attempt_num in range(1, max_citation_attempts + 1):
                logger.info(f"--- Citation SQL Attempt {attempt_num}/{max_citation_attempts} ---")

                citation_handler = LoggingHandler(
                    prompt_file="sliders/inspect_answer/query_generator.prompt",
                    metadata={
                        "question": question,
                        "citation_mode": True,
                        "question_id": metadata.get("question_id", None),
                        "stage": f"citation_sql_attempt_{attempt_num}",
                    },
                )

                # Format citation attempts history
                citation_attempts_history = "No previous attempts yet"
                if citation_attempts:
                    history_parts = []
                    for idx, (sql, result_preview, error_msg) in enumerate(citation_attempts):
                        history_parts.append(
                            f"### Attempt {idx + 1}\n**SQL:**\n```sql\n{sql}\n```\n**Result:** {result_preview}\n**Error:** {error_msg or 'None'}"
                        )
                    citation_attempts_history = "\n\n".join(history_parts)

                citation_decision = await citation_chain.ainvoke(
                    {
                        "question": question,
                        "schema": schema,
                        "table_stats": table_stats_list,
                        "query_history": citation_query_history,
                        "finalization_reasoning": finalization_reasoning,
                        "final_answer": final_answer if isinstance(final_answer, str) else final_answer.content,
                        "citation_mode": True,
                        "citation_attempts_history": citation_attempts_history,
                        "attempts_remaining": max_citation_attempts - attempt_num,
                        "strategy": "",  # Not needed in citation mode
                        "queries_remaining": 0,  # Not needed in citation mode
                        "row_limit": 0,  # Not needed in citation mode
                        "is_first_query": False,  # Not needed in citation mode
                        "sql_error_feedback": "No previous SQL errors",  # Not needed in citation mode
                    },
                    config={"callbacks": [citation_handler]},
                )

                logger.info(f"Citation action: {citation_decision.action}")
                logger.info(f"Citation SQL reasoning: {citation_decision.reasoning}")

                if citation_decision.action == "finalize":
                    logger.info("✓ Agent finalized citation SQL")
                    # Use the last successful result
                    if citation_result is not None:
                        logger.info(f"Using previous citation result: {len(citation_result)} rows")
                        final_citation_sql = citation_attempts[-1][0] if citation_attempts else None
                        final_citation_reasoning = citation_decision.reasoning
                        break
                    else:
                        logger.warning("Agent finalized but no successful citation result available")
                        break

                # action == "execute": try the SQL
                logger.info(f"Citation SQL query:\n{citation_decision.sql}")

                try:
                    result_df, error = run_sql_query(citation_decision.sql, duck_sql_conn, output_format="dataframe")
                    if error:
                        error_msg = str(result_df)
                        logger.error(f"✗ Citation SQL execution failed: {error_msg}")
                        citation_attempts.append((citation_decision.sql, "SQL Error", error_msg))
                    else:
                        logger.info(f"✓ Citation SQL executed successfully: {len(result_df)} rows retrieved")
                        # Format result preview
                        result_preview = format_sql_result(result_df, max_rows=5)
                        citation_attempts.append((citation_decision.sql, result_preview, None))
                        # Store successful result
                        citation_result = result_df
                        final_citation_sql = citation_decision.sql
                        final_citation_reasoning = citation_decision.reasoning
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"✗ Citation SQL execution exception: {error_msg}")
                    citation_attempts.append((citation_decision.sql, "Exception", error_msg))

            # After loop, check if we have a successful result
            if citation_result is not None and final_citation_sql is not None:
                logger.info(f"✓ Citation SQL finalized: {len(citation_result)} rows")

                # Store citation information in metadata
                if "answer_generation" not in metadata:
                    metadata["answer_generation"] = {}
                metadata["answer_generation"]["citation_sql"] = final_citation_sql
                metadata["answer_generation"]["citation_rows"] = len(citation_result)
                metadata["answer_generation"]["citation_reasoning"] = final_citation_reasoning
                metadata["answer_generation"]["citation_attempts"] = len(citation_attempts)

                # Check if citation SQL returned empty rows
                if len(citation_result) == 0:
                    logger.info("Citation SQL returned 0 rows, skipping citation paragraph generation")
                    metadata["answer_generation"]["citation_paragraph"] = None
                else:
                    # ========================================================================
                    # GENERATE CITATION PARAGRAPH
                    # ========================================================================
                    logger.info("=== Citation Paragraph Generation ===")

                    # Format citation data
                    formatted_citation_data = format_sql_result(citation_result, max_rows=20)

                    # Create citation verbalization chain
                    from langchain_core.output_parsers import StrOutputParser

                    citation_verbalize_template = load_fewshot_prompt_template(
                        template_file="sliders/inspect_answer/verbalize_results.prompt",
                        template_blocks=[],
                    )
                    citation_verbalize_chain = citation_verbalize_template | llm_client | StrOutputParser()

                    # Generate citation paragraph using verbalization chain
                    citation_verbalize_handler = LoggingHandler(
                        prompt_file="sliders/inspect_answer/verbalize_results.prompt",
                        metadata={
                            "question": question,
                            "citation_mode": True,
                            "question_id": metadata.get("question_id", None),
                            "stage": "citation_verbalize",
                        },
                    )

                    citation_paragraph = await citation_verbalize_chain.ainvoke(
                        {
                            "question": question,
                            "classes": prepare_schema_for_template(schema),
                            "citation_mode": True,
                            "generated_answer": final_answer if isinstance(final_answer, str) else final_answer.content,
                            "citation_sql": final_citation_sql,
                            "citation_data": formatted_citation_data,
                            "tool_call": "",  # Not needed in citation mode
                            "tool_output": "",  # Not needed in citation mode
                        },
                        config={"callbacks": [citation_verbalize_handler]},
                    )

                    logger.info("✓ Citation paragraph generated")

                    # Store citation paragraph in metadata (not appended to final answer for evaluation)
                    metadata["answer_generation"]["citation_paragraph"] = citation_paragraph
            else:
                # No successful citation result after all attempts
                logger.warning(f"✗ No successful citation SQL after {max_citation_attempts} attempts")
                if "answer_generation" not in metadata:
                    metadata["answer_generation"] = {}
                metadata["answer_generation"]["citation_sql_error"] = (
                    f"No successful citation after {len(citation_attempts)} attempts"
                )

        except Exception as e:
            logger.error(f"✗ Citation SQL generation failed: {e}")
            if "answer_generation" not in metadata:
                metadata["answer_generation"] = {}
            metadata["answer_generation"]["citation_generation_error"] = str(e)
    else:
        logger.info("No query history, skipping citation SQL generation")

    # ========================================================================
    # VERBALIZE RECONCILIATION STATISTICS
    # ========================================================================
    if enable_recon_stats:
        logger.info("=== Reconciliation Statistics Verbalization ===")
    else:
        logger.info("Reconciliation stats verbalization disabled, skipping")

    if enable_recon_stats and reconciliation_stats:
        try:
            # Create reconciliation stats verbalization chain with dedicated prompt
            from langchain_core.output_parsers import StrOutputParser
            import json

            recon_verbalize_template = load_fewshot_prompt_template(
                template_file="sliders/reconcilation/verbalize_stats.prompt",
                template_blocks=[],
            )
            recon_verbalize_chain = recon_verbalize_template | llm_client | StrOutputParser()

            recon_verbalize_handler = LoggingHandler(
                prompt_file="sliders/reconcilation/verbalize_stats.prompt",
                metadata={
                    "question": question,
                    "question_id": metadata.get("question_id", None),
                    "stage": "reconciliation_stats_verbalize",
                },
            )

            reconciliation_stats_summary = await recon_verbalize_chain.ainvoke(
                {
                    "question": question,
                    "classes": prepare_schema_for_template(schema),
                    "reconciliation_stats": json.dumps(reconciliation_stats, indent=2),
                },
                config={"callbacks": [recon_verbalize_handler]},
            )

            logger.info("✓ Reconciliation statistics summary generated")

            # Store reconciliation stats summary in metadata
            if "answer_generation" not in metadata:
                metadata["answer_generation"] = {}
            metadata["answer_generation"]["reconciliation_stats_summary"] = reconciliation_stats_summary

        except Exception as e:
            logger.error(f"✗ Reconciliation stats verbalization failed: {e}")
            if "answer_generation" not in metadata:
                metadata["answer_generation"] = {}
            metadata["answer_generation"]["reconciliation_stats_verbalization_error"] = str(e)
    else:
        logger.info("No reconciliation stats available, skipping reconciliation stats verbalization")

    # Record timing
    answer_time = __import__("time").time() - answer_start_time
    if "answer_generation" not in metadata:
        metadata["answer_generation"] = {}
    metadata["answer_generation"]["answer_time"] = answer_time
    metadata["answer_generation"]["num_queries"] = len(query_history)

    if isinstance(final_answer, str):
        metadata["answer_generation"]["final_answer_tokens"] = len(final_answer.split())

    logger.info(f"=== Completed in {answer_time:.2f}s with {len(query_history)} queries ===")

    return final_answer
