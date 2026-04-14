from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from pydantic import BaseModel
import pandas as pd
import duckdb
import warnings

from sliders.log_utils import logger

if TYPE_CHECKING:
    from sliders.llm_models import Tables, Field


def string_to_type(value_type: str) -> Type:
    # Add typing module types to the local namespace
    local_namespace = {
        "List": List,
        "list": list,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Literal": Literal,
    }

    # Handle special case for Literal types
    original_value_type = value_type
    value_type = value_type.strip()

    if value_type.lower().startswith("literal["):
        # Extract the arguments part and reconstruct with proper capitalization
        args_part = value_type[value_type.find("[") :]
        value_type = f"Literal{args_part}"

    try:
        # Evaluate the string safely using eval
        return eval(value_type, {"__builtins__": None}, local_namespace)
    except Exception as e:
        logger.warning(f"Failed to parse type: {original_value_type} -> {value_type}, error: {e}")
        try:
            # If eval fails with the modified string, try with the original
            return eval(original_value_type, {"__builtins__": None}, local_namespace)
        except Exception as e2:
            raise ValueError(f"Invalid type string: {original_value_type}") from e2


def type_to_str(tp):
    """Convert a type annotation to string, handling containers and Pydantic models."""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            # Recurse into nested BaseModel
            fields = ", ".join(f"{k}: {type_to_str(v.annotation)}" for k, v in tp.model_fields.items())
            return f"{tp.__name__}({fields})"
        return tp.__name__ if hasattr(tp, "__name__") else str(tp)

    elif origin in (list, List):
        return f"List[{type_to_str(args[0])}]"
    elif origin in (dict, Dict):
        return f"Dict[{type_to_str(args[0])}, {type_to_str(args[1])}]"
    elif origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) < len(args):
            return f"Optional[{type_to_str(non_none[0])}]"
        return "Union[" + ", ".join(type_to_str(a) for a in args) + "]"
    else:
        return str(tp)


def pydantic_model_to_signature(model: type[BaseModel], func_name: str = "my_function") -> str:
    type_hints = get_type_hints(model)
    args = []

    for name, field in model.model_fields.items():
        annotation = type_hints.get(name, Any)
        type_str = type_to_str(annotation)

        if field.is_required:
            args.append(f"{name}: {type_str}")
        else:
            default_repr = repr(field.default)
            args.append(f"{name}: {type_str} = {default_repr}")

    return f"def {func_name}({', '.join(args)}):\n    pass"


def tables_to_template_dicts(tables: list) -> list[str]:
    """Convert ExtractedTable objects to pre-formatted strings for Jinja2 templates.

    LangChain's Jinja2 sandbox restricts ALL attribute/key access (even on dicts),
    so we pre-format each table as a complete string.
    """
    formatted = []
    for t in tables:
        if hasattr(t, "dataframe_table_name") and hasattr(t, "table_str"):
            formatted.append(f'## Table Name: "{t.dataframe_table_name}"\n{t.table_str}')
        elif isinstance(t, dict):
            name = t.get("dataframe_table_name", "unknown")
            content = t.get("table_str", "")
            formatted.append(f'## Table Name: "{name}"\n{content}')
        else:
            formatted.append(str(t))
    return formatted


def prepare_schema_repr(schema: "Tables") -> str:
    """Prepare schema representation as a formatted string (for human-readable prompts)."""
    class_repr = ""
    for one_table in schema.tables:
        one_class_repr = f"## Table Name: {one_table.name} ({one_table.description})\n"
        for field in one_table.fields:
            one_class_repr += f"### Field Name: {field.name}\n"
            one_class_repr += f"Description: {field.description}\n"
            one_class_repr += f"Data Type: {field.data_type}\n"
            if field.data_type == "enum" or field.enum_values is not None:
                one_class_repr += f"Enum Values: {field.enum_values}\n"
            one_class_repr += f"Unit: {field.unit}\n"
            one_class_repr += f"Scale: {field.scale}\n"
            # one_class_repr += f"Required: {field.required}\n"
            one_class_repr += f"Normalization: {field.normalization}\n"
        class_repr += one_class_repr + "\n"
    return class_repr


def get_table_schema(table_name: str, schema: Tables) -> list[Field]:
    for table in schema.tables:
        if table.name == table_name:
            return table.fields
    return None


def format_fields_for_template(fields: list) -> str:
    """
    Format a list of Field objects or dicts as a string for use in Jinja2 templates.

    LangChain's Jinja2 sandbox restricts attribute access, so we pre-format
    the fields as a string instead of passing objects.
    """
    if not fields:
        return ""

    def get_field_attr(field, attr, default=""):
        """Get attribute from Field object or dict."""
        if isinstance(field, dict):
            return field.get(attr, default)
        return getattr(field, attr, default)

    parts = []
    for field in fields:
        name = get_field_attr(field, "name", "unknown")
        description = get_field_attr(field, "description", "")
        data_type = get_field_attr(field, "data_type", "")
        unit = get_field_attr(field, "unit", "")
        parts.append(f"## Field\nName: {name}\nDescription: {description}\nData Type: {data_type}\nUnit: {unit}")
    return "\n\n".join(parts)


def sanitize_table_name(name: str) -> str:
    # Replace special characters and spaces with underscore
    import re

    # Remove all special characters except alphanumeric and underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def format_table(table_data):
    # No columns → nothing to format
    if table_data.shape[1] == 0:
        return None

    header = tuple(table_data.columns.to_list())
    records = table_data.to_records(index=False)

    return str(header) + "\n" + "\n".join(str(row) for row in records)


def prepare_schema_dict(schema: "Tables") -> list:
    """
    Prepare schema representation as a list of dicts (for structured use in prompts).

    Note: This is different from prepare_schema_repr() which returns a formatted string.
    """
    schema_list = []
    for table in schema.tables:
        table_dict = {"name": table.name, "description": table.description, "fields": []}
        for field in table.fields:
            field_dict = {
                "name": field.name,
                "description": field.description,
                "unit": field.unit if hasattr(field, "unit") else "",
            }
            table_dict["fields"].append(field_dict)
        schema_list.append(table_dict)
    return schema_list


def prepare_schema_for_template(schema: "Tables") -> str:
    """
    Format schema as a string for use in Jinja2 templates.

    LangChain's Jinja2 sandbox restricts attribute access, so we pre-format
    the schema as a string instead of passing objects/dicts.
    """
    parts = []
    for table in schema.tables:
        table_part = f"### Table: {table.name}\n{table.description}\n\nFields:"
        for field in table.fields:
            unit = field.unit if hasattr(field, "unit") and field.unit else "no unit"
            table_part += f"\n- **{field.name}** ({unit}): {field.description}"
        parts.append(table_part)
    return "\n\n".join(parts)


def prepare_table_stats_for_template(table_stats: list[dict]) -> str:
    """
    Format table statistics as a string for use in Jinja2 templates.

    LangChain's Jinja2 sandbox restricts attribute access, so we pre-format
    the stats as a string instead of passing dicts.
    """
    parts = []
    for stat in table_stats:
        table_name = stat.get("table_name", "unknown")
        stats = stat.get("stats", "")
        parts.append(f"### Table: {table_name}\n{stats}")
    return "\n\n".join(parts)


def get_schema_date_columns(schema: "Tables" | None, table_name: str) -> list[str]:
    """Return base field names marked as date in the schema for the given table."""
    if not schema or not getattr(schema, "tables", None):
        return []

    try:
        table = next(t for t in schema.tables if t.name == table_name)
    except StopIteration:
        return []

    date_fields: list[str] = []
    for field in table.fields:
        data_type = getattr(field, "data_type", "") or ""
        if isinstance(data_type, str) and data_type.lower() == "date":
            date_fields.append(field.name)
    return date_fields


def get_schema_date_format_map(schema: "Tables" | None, table_name: str) -> dict[str, str]:
    """Return mapping of date column -> expected date_format from schema normalization."""
    if not schema or not getattr(schema, "tables", None):
        return {}

    try:
        table = next(t for t in schema.tables if t.name == table_name)
    except StopIteration:
        return {}

    fmt_map: dict[str, str] = {}
    for field in table.fields:
        data_type = getattr(field, "data_type", "") or ""
        normalization = getattr(field, "normalization", None)
        if isinstance(data_type, str) and data_type.lower() == "date":
            if normalization and getattr(normalization, "date_format", None):
                fmt_map[field.name] = normalization.date_format
    return fmt_map


def convert_schema_date_format_to_strftime(schema_format: str) -> str:
    """
    Convert schema date format (e.g., 'YYYY-MM-DD') to Python strftime format (e.g., '%Y-%m-%d').

    Common schema format tokens:
        YYYY -> %Y (4-digit year)
        YY   -> %y (2-digit year)
        MM   -> %m (2-digit month)
        DD   -> %d (2-digit day)
        HH   -> %H (24-hour)
        mm   -> %M (minute)
        ss   -> %S (second)
    """
    if not schema_format:
        return None

    # Order matters - longer tokens first to avoid partial replacements
    replacements = [
        ("YYYY", "%Y"),
        ("YY", "%y"),
        ("MM", "%m"),
        ("DD", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    ]

    result = schema_format
    for schema_token, strftime_token in replacements:
        result = result.replace(schema_token, strftime_token)

    return result


def coerce_date_columns(
    df: pd.DataFrame,
    *,
    date_columns: Iterable[str] | None = None,
    date_formats: dict[str, str] | None = None,
    parse_success_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Convert date-like columns to python ``date`` objects so DuckDB registers them as DATE.

    Heuristics:
    - Explicit column names passed via ``date_columns`` are always attempted.
    - Columns whose name contains "date" are considered date candidates.
    - For datetime-like dtypes, drop the time component.
    - For object/string columns, attempt ``pd.to_datetime`` (using schema date_format when available)
      and only convert when the
      parsed success ratio meets ``parse_success_threshold`` to avoid false positives.

    Args:
        date_formats: dict mapping column names to schema date formats (e.g., "YYYY-MM-DD").
                      These are automatically converted to strftime format for pandas.
    """
    if df is None or df.empty:
        return df

    coerced = df.copy()
    date_formats = date_formats or {}
    candidate_cols = set(date_columns or [])
    candidate_cols.update(col for col in coerced.columns if "date" in str(col).lower())
    candidate_cols.update(date_formats.keys())

    for col in candidate_cols:
        if col not in coerced.columns:
            continue

        series = coerced[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            coerced[col] = series.dt.date
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Convert schema format (e.g., "YYYY-MM-DD") to strftime format (e.g., "%Y-%m-%d")
            schema_fmt = date_formats.get(col)
            strftime_fmt = convert_schema_date_format_to_strftime(schema_fmt) if schema_fmt else None

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Could not infer format")
                    parsed = (
                        pd.to_datetime(series, format=strftime_fmt, errors="coerce")
                        if strftime_fmt
                        else pd.to_datetime(series, errors="coerce")
                    )
            except Exception:
                continue

            parse_ratio = parsed.notna().mean() if len(parsed) else 0
            if parse_ratio >= parse_success_threshold and parse_ratio > 0:
                coerced[col] = parsed.dt.date

    return coerced


def register_df_with_duckdb(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    name: str,
    *,
    date_columns: Iterable[str] | None = None,
    schema: "Tables" | None = None,
    schema_table_name: str | None = None,
) -> pd.DataFrame:
    """
    Register a DataFrame with DuckDB after ensuring date columns are typed as DATE.

    Returns the DataFrame that was registered (potentially with coerced date columns).
    """
    schema_date_cols = get_schema_date_columns(schema, schema_table_name or name)
    schema_date_formats = get_schema_date_format_map(schema, schema_table_name or name)
    combined_date_columns = set(date_columns or [])
    combined_date_columns.update(schema_date_cols)
    combined_date_columns.update(schema_date_formats.keys())

    coerced_df = coerce_date_columns(
        df,
        date_columns=combined_date_columns,
        date_formats=schema_date_formats,
    )

    tmp_name = f"__tmp_{name}"
    conn.register(tmp_name, coerced_df)

    # Build SELECT with explicit DATE casts for schema-declared date columns
    select_exprs = []
    for col in coerced_df.columns:
        if col in combined_date_columns:
            # Robustly coerce to DATE:
            # - Cast to VARCHAR so TRIM/NULLIF always work even if column is already DATE
            # - TRIM to remove whitespace, NULLIF to turn empty strings into NULL
            # - TRY_CAST to avoid raising on unparseable values (returns NULL instead)
            select_exprs.append(f'TRY_CAST(NULLIF(TRIM(CAST("{col}" AS VARCHAR)), \'\') AS DATE) AS "{col}"')
        else:
            select_exprs.append(f'"{col}"')

    select_sql = ", ".join(select_exprs)
    conn.execute(f'CREATE OR REPLACE TEMP TABLE "{name}" AS SELECT {select_sql} FROM "{tmp_name}"')
    conn.unregister(tmp_name)

    # Return the DataFrame that matches the registered table contents
    return conn.execute(f'SELECT * FROM "{name}"').fetchdf()


def get_table_stats(df: pd.DataFrame, table_name: str) -> dict:
    """Get summary statistics about the table without loading full data."""
    conn = duckdb.connect()
    register_df_with_duckdb(conn, df, table_name)

    # Get basic stats
    row_count = conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()[0]

    # Get column info
    columns_info = []
    for col in df.columns:
        # Quote column names to handle special characters like colons
        quoted_col = f'"{col}"'
        null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {quoted_col} IS NULL").fetchone()[0]
        distinct_count = conn.execute(f"SELECT COUNT(DISTINCT {quoted_col}) FROM {table_name}").fetchone()[0]
        columns_info.append(
            {"name": col, "dtype": str(df[col].dtype), "null_count": null_count, "distinct_count": distinct_count}
        )

    conn.close()

    return {"row_count": row_count, "column_count": len(df.columns), "columns": columns_info}


def format_table_stats(stats: dict) -> str:
    """Format table statistics for display in prompts."""
    result = "Table Statistics:\n"
    result += f"- Total rows: {stats['row_count']}\n"
    result += f"- Total columns: {stats['column_count']}\n\n"
    result += "Column Details:\n"
    for col in stats["columns"]:
        result += (
            f"  - {col['name']} ({col['dtype']}): {col['null_count']} nulls, {col['distinct_count']} distinct values\n"
        )
    return result


def format_sql_result(result: pd.DataFrame, max_rows: int = 10) -> str:
    """Format SQL query result for display in prompts."""
    if result is None or result.empty:
        return "No results"

    if len(result) > max_rows:
        return f"Shape: {result.shape}\nFirst {max_rows} rows:\n" + result.head(max_rows).to_string(index=False)
    return f"Shape: {result.shape}\n" + result.to_string(index=False)


def format_dataframe_schema(df: pd.DataFrame) -> str:
    """Format DataFrame schema for display in prompts."""
    if df is None or df.empty:
        return "Empty DataFrame - no schema available"

    schema_lines = [f"Columns: {len(df.columns)}", f"Rows: {len(df)}", ""]
    schema_lines.append("Column Name | Data Type")
    schema_lines.append("-" * 50)

    for col in df.columns:
        dtype = str(df[col].dtype)
        schema_lines.append(f"{col} | {dtype}")

    return "\n".join(schema_lines)


def format_table_schema(schema: "Tables", table_name: str) -> str:
    """Format the schema for a specific table for display in prompts."""
    schema_list = prepare_schema_dict(schema)

    for table_schema in schema_list:
        if table_schema.get("name") == table_name:
            table_description = table_schema.get("description", "")
            fields_desc = []
            for field in table_schema.get("fields", []):
                field_info = f"  - {field['name']}"
                if field.get("description"):
                    field_info += f": {field['description']}"
                if field.get("unit"):
                    field_info += f" [unit: {field['unit']}]"
                fields_desc.append(field_info)

            header = f"Table: {table_name}"
            if table_description:
                header += f"\nDescription: {table_description}"
            return header + "\nFields:\n" + "\n".join(fields_desc)

    return f"Table: {table_name}\n(Schema not found)"


def build_pk_filter_sql(pk_cols: list[str], pk_values: tuple) -> str:
    """Build WHERE clause for primary key filtering."""
    conditions = []
    for col, val in zip(pk_cols, pk_values):
        if pd.isna(val):
            conditions.append(f'"{col}" IS NULL')
        else:
            # Escape single quotes
            escaped = str(val).replace("'", "''")
            conditions.append(f"\"{col}\" = '{escaped}'")
    return " AND ".join(conditions)


def run_sql(sql: str, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Execute SQL query on a dataframe using DuckDB."""
    conn = duckdb.connect()
    conn.register(table_name, df)
    result = conn.execute(sql).fetchdf()
    conn.close()
    return result
