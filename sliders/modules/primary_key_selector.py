"""Primary key selection module for automatically extracted tables."""

from collections import Counter
from typing import Literal, Optional
from pydantic import BaseModel

from sliders.llm_models import Tables
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.llm_tools.sql import run_sql_query
from sliders.utils import get_table_stats, format_table_stats, format_table_schema


class PrimaryKeyDecision(BaseModel):
    """Decision from the primary key selector: query for more info or finalize selection."""

    reasoning: str
    action: Literal["query", "finalize"]
    sql: Optional[str] = None  # SQL query if action is "query"
    primary_key: Optional[list[str]] = None  # List of field names if action is "finalize"


class PrimaryKeySelector:
    """Selects appropriate primary keys for extracted tables using LLM + SQL analysis."""

    def __init__(self, model_config: dict, pk_selection_config: dict = None):
        """
        Initialize the primary key selector.

        Args:
            model_config: Model configuration dict with keys for different LLM tasks
            pk_selection_config: Configuration for primary key selection behavior (queries, retries, etc.)
        """
        self.model_config = model_config
        self.pk_selection_config = pk_selection_config or {}

    async def select_primary_keys(
        self,
        question: str,
        tables: list,
        schema: Tables,
        duck_sql_conn,
        metadata: dict,
    ) -> dict:
        """
        Select primary keys for all tables.

        Args:
            question: The original question being answered
            tables: List of Table objects with dataframes
            schema: Schema information
            duck_sql_conn: DuckSQLBasic connection with registered tables
            metadata: Metadata dictionary for logging

        Returns:
            Dictionary mapping table_name -> {
                "primary_key": list[str],
                "reasoning": str,
                "query_count": int
            }
        """
        primary_key_selections = {}

        for table in tables:
            if table.dataframe is None or table.dataframe.empty:
                logger.info(f"Skipping primary key selection for empty table: {table.name}")
                continue

            logger.info(f"=== Selecting Primary Key for {table.name} ===")

            selection = await self._select_primary_key_for_table(
                question=question,
                table=table,
                schema=schema,
                duck_sql_conn=duck_sql_conn,
                metadata=metadata,
            )

            primary_key_selections[table.name] = selection

            logger.info(f"Selected primary key for {table.name}: {selection['primary_key']}")
            logger.info(f"Reasoning: {selection['reasoning']}")

        return primary_key_selections

    async def _select_primary_key_for_table(
        self,
        question: str,
        table,
        schema: Tables,
        duck_sql_conn,
        metadata: dict,
    ) -> dict:
        """
        Select primary key for a single table using majority voting over k runs.

        Returns:
            dict with keys: primary_key (list[str]), reasoning (str), query_count (int),
                           voting_results (dict) if voting_k > 1
        """
        voting_k = self.pk_selection_config.get("voting_k", 3)

        if voting_k <= 1:
            # No voting, just run once
            return await self._select_primary_key_single_run(
                question=question,
                table=table,
                schema=schema,
                duck_sql_conn=duck_sql_conn,
                metadata=metadata,
                run_number=1,
            )

        # Run k times and collect results
        logger.info(f"=== Running {voting_k} voting rounds for primary key selection ===")
        all_results = []

        for k in range(1, voting_k + 1):
            logger.info(f"\n--- Voting Round {k}/{voting_k} ---")
            result = await self._select_primary_key_single_run(
                question=question,
                table=table,
                schema=schema,
                duck_sql_conn=duck_sql_conn,
                metadata=metadata,
                run_number=k,
            )
            all_results.append(result)
            logger.info(f"  Round {k} selected: {result['primary_key']}")

        # Majority voting on the complete primary key set
        pk_tuples = [tuple(r["primary_key"]) for r in all_results]
        pk_counts = Counter(pk_tuples)
        most_common_pk, vote_count = pk_counts.most_common(1)[0]
        winning_pk = list(most_common_pk)

        # Collect reasoning from runs that selected the winning PK
        winning_reasonings = [r["reasoning"] for r in all_results if tuple(r["primary_key"]) == most_common_pk]
        combined_reasoning = winning_reasonings[0] if winning_reasonings else "Majority vote"

        # Calculate total queries across all runs
        total_queries = sum(r["query_count"] for r in all_results)

        logger.info("\n=== Majority Voting Result ===")
        logger.info(f"  Winning Primary Key: {winning_pk}")
        logger.info(f"  Votes: {vote_count}/{voting_k}")
        logger.info(f"  Vote Distribution: {dict(pk_counts)}")

        return {
            "primary_key": winning_pk,
            "reasoning": combined_reasoning,
            "query_count": total_queries,
            "voting_results": {
                "voting_k": voting_k,
                "vote_count": vote_count,
                "distribution": {str(list(pk)): count for pk, count in pk_counts.items()},
                "all_selections": [list(pk) for pk in pk_tuples],
            },
        }

    async def _select_primary_key_single_run(
        self,
        question: str,
        table,
        schema: Tables,
        duck_sql_conn,
        metadata: dict,
        run_number: int = 1,
    ) -> dict:
        """
        Select primary key for a single table using iterative SQL inspection (single run).

        Returns:
            dict with keys: primary_key (list[str]), reasoning (str), query_count (int)
        """
        # Get table statistics
        df = table.dataframe
        table_name = table.dataframe_table_name
        stats = get_table_stats(df, table_name)

        # Initialize LLM chain
        pk_model_config = self.model_config.get("select_primary_key", {})
        model = pk_model_config.get("model", "gpt-4.1")
        temperature = pk_model_config.get("temperature", 0.7)  # Default 0.7 for voting variance
        llm_client = get_llm_client(model=model, temperature=temperature)

        query_template = load_fewshot_prompt_template(
            template_file="sliders/select_primary_key.prompt",
            template_blocks=[],
        )
        query_chain = query_template | llm_client.with_structured_output(PrimaryKeyDecision)

        # Query loop
        max_queries = self.pk_selection_config.get("max_queries", 5)
        max_retries = self.pk_selection_config.get("max_retries", 3)
        query_history = []  # List of (reasoning, sql, result) tuples

        for query_num in range(max_queries):
            queries_remaining = max_queries - query_num
            logger.info(f"--- Primary Key Analysis Query {query_num + 1}/{max_queries} (Run {run_number}) ---")

            # Try to generate and execute a query with retries
            query_attempts = []
            sql_error_feedback = None
            query_successful = False

            for retry in range(max_retries):
                if retry > 0:
                    logger.info(f"  Retry {retry + 1}/{max_retries}")

                handler = LoggingHandler(
                    prompt_file="sliders/select_primary_key.prompt",
                    metadata={
                        "question": question,
                        "table_name": table_name,
                        "stage": f"primary_key_run_{run_number}_query_{query_num + 1}_attempt_{retry + 1}",
                        "question_id": metadata.get("question_id", None),
                        "voting_run": run_number,
                        **(metadata or {}),
                    },
                )

                # Format query history
                formatted_history = []
                for idx, (reasoning, sql, result) in enumerate(query_history):
                    formatted_history.append(
                        f"Query {idx + 1}:\nReasoning: {reasoning}\nSQL: {sql}\nResult:\n{result}\n"
                    )

                # Prepare error feedback if this is a retry
                if query_attempts:
                    feedback_parts = []
                    for idx, (sql, err) in enumerate(query_attempts):
                        feedback_parts.append(f"Attempt {idx + 1} failed:\nSQL: {sql}\nError: {err}\n")
                    sql_error_feedback = (
                        "\n".join(feedback_parts) + f"\nThis is attempt {retry + 1}/{max_retries}. Please fix the SQL."
                    )

                # Get primary key decision
                decision = await query_chain.ainvoke(
                    {
                        "question": question,
                        "table_name": table_name,
                        "schema": format_table_schema(schema, table.name),
                        "table_stats": format_table_stats(stats),
                        "query_history": "\n".join(formatted_history)
                        if formatted_history
                        else "No queries executed yet",
                        "queries_remaining": queries_remaining,
                        "sql_error_feedback": sql_error_feedback or "No previous SQL errors",
                        "max_queries": max_queries,
                    },
                    config={"callbacks": [handler]},
                )

                if decision.action == "finalize":
                    logger.info(f"✓ Primary key selected after {len(query_history)} queries (Run {run_number})")
                    logger.info(f"  Primary Key: {decision.primary_key}")
                    logger.info(f"  Reasoning: {decision.reasoning[:120]}...")
                    return {
                        "primary_key": decision.primary_key or [],
                        "reasoning": decision.reasoning,
                        "query_count": len(query_history),
                    }

                # Execute query
                if decision.sql:
                    logger.info(f"Running SQL: {decision.sql[:100]}{'...' if len(decision.sql) > 100 else ''}")
                    result, error = run_sql_query(decision.sql, duck_sql_conn, output_format="formatted")

                    if not error:
                        # Success - add to history
                        query_history.append((decision.reasoning, decision.sql, result))
                        logger.info("✓ Query successful")
                        query_successful = True
                        break
                    else:
                        # Error - add to attempts and retry
                        logger.warning(f"  → SQL Error: {result}")
                        query_attempts.append((decision.sql, result))

                        if retry == max_retries - 1:
                            logger.error(f"✗ All {max_retries} attempts failed")
                else:
                    logger.warning("Agent chose to query but provided no SQL")
                    break

            # After retry loop, check if we finalized
            if decision.action == "finalize":
                return {
                    "primary_key": decision.primary_key or [],
                    "reasoning": decision.reasoning,
                    "query_count": len(query_history),
                }

            if not query_successful:
                logger.warning(f"Query {query_num + 1} failed, moving to next query (Run {run_number})")

        # If we exhausted all queries without finalizing, return a default
        logger.warning(f"Exhausted query budget for {table_name}, using fallback primary key (Run {run_number})")
        return {
            "primary_key": ["__document_name__", "__chunk_id__"],
            "reasoning": "Query budget exhausted. Using synthetic key based on extraction metadata.",
            "query_count": len(query_history),
        }
