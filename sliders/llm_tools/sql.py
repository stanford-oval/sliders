import duckdb
import pandas as pd

from sliders.utils import register_df_with_duckdb


class DuckSQLBasic:
    def __init__(self, db=":memory:"):
        self.conn = duckdb.connect(db)

    def __enter__(self):  # allow `with DuckSQLBasic() as db: …`
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.close()

    def sql(self, query: str):
        return self.conn.sql(query)

    def register(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        date_columns: list[str] | None = None,
        schema=None,
        schema_table_name: str | None = None,
    ):
        register_df_with_duckdb(
            self.conn,
            df,
            name,
            date_columns=date_columns,
            schema=schema,
            schema_table_name=schema_table_name,
        )

    def unregister(self, name: str):
        self.conn.unregister(name)

    def close(self):
        self.conn.close()


class DuckSQL:
    con = None

    def __init__(self, id_column="id", value_column="value"):
        if DuckSQL.con is None:
            DuckSQL.con = duckdb.connect()
        self.con = DuckSQL.con
        DuckSQL._row_id_column = id_column
        DuckSQL._row_value_column = value_column

        # Register UDFs that only look at our in-memory cache
        self.con.create_function(
            name="filter_by",
            function=DuckSQL.choose,
            parameters=[duckdb.typing.BIGINT, duckdb.typing.VARCHAR],
            return_type=duckdb.typing.BOOLEAN,
            side_effects=True,
        )
        self.con.create_function(
            name="cluster_on",
            function=DuckSQL.assign_cluster,
            parameters=[duckdb.typing.BIGINT, duckdb.typing.VARCHAR],
            return_type=duckdb.typing.VARCHAR,
            side_effects=True,
        )

    def register(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        date_columns: list[str] | None = None,
        schema=None,
        schema_table_name: str | None = None,
    ):
        # Cache the full table once, then register for SQL
        registered_df = register_df_with_duckdb(
            self.con,
            df,
            name,
            date_columns=date_columns,
            schema=schema,
            schema_table_name=schema_table_name,
        )
        DuckSQL._rows = registered_df.to_dict("records")

    def sql(self, query: str):
        return self.con.sql(query)


def run_sql_query(query: str, duck_sql_conn: DuckSQLBasic, row_limit: int = None, output_format: str = "markdown"):
    """
    Run a SQL query on the database.

    Args:
        query: SQL query to execute
        duck_sql_conn: DuckSQLBasic connection with registered tables
        row_limit: Maximum rows to display in output (None = all rows)
        output_format: Format of output - "markdown", "formatted", or "dataframe"

    Returns:
        Tuple of (result, error_flag)
        - If output_format="dataframe", returns (DataFrame, error_flag)
        - Otherwise, returns (result_string, error_flag)
    """
    try:
        result = duck_sql_conn.sql(query).to_df()

        if result is None or result.empty:
            if output_format == "dataframe":
                return result, False
            return "No results", False

        # Return raw DataFrame if requested
        if output_format == "dataframe":
            return result, False

        # Apply row limit if specified
        if row_limit is not None and len(result) > row_limit:
            display_df = result.head(row_limit)
            if output_format == "formatted":
                formatted = f"Shape: {result.shape}\nFirst {row_limit} rows:\n" + display_df.to_string(index=False)
            else:  # markdown
                formatted = f"Shape: {result.shape}\nFirst {row_limit} rows:\n" + display_df.to_markdown()
        else:
            if output_format == "formatted":
                formatted = f"Shape: {result.shape}\n" + result.to_string(index=False)
            else:  # markdown
                formatted = result.to_markdown()

        return formatted, False
    except Exception as e:
        return f"Error running SQL query: {query}\n{e}. Please try again.", True
