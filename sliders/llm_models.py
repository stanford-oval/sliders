from typing import Any, Literal, Optional, Type, get_args, get_origin
from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField

from sliders.log_utils import logger
from sliders.utils import pydantic_model_to_signature, string_to_type


class IsRelevantPage(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision if the page is relevant to the question.",
    )
    is_relevant: bool = PydanticField(
        ...,
        description="Whether the page is relevant to the question.",
    )


class InformationDensityResponse(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Explanation of why the chunk is or is not information-dense based on the schema requirements.",
    )
    estimated_row_count: int = PydanticField(
        ...,
        description="Estimated number of distinct rows/items that need to be extracted from this chunk.",
        ge=0,
    )
    is_dense: bool = PydanticField(
        ...,
        description="True if the chunk contains multiple rows worth of data that would benefit from row-by-row extraction.",
    )


class RowExtractionInstructions(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Explanation of how the chunk was divided into distinct extractable items.",
    )
    row_descriptions: list[str] = PydanticField(
        ...,
        description="List of descriptive instructions for extracting each row/item separately. Each instruction should clearly identify which specific row or item to extract.",
        min_items=1,
    )


class Evaluation(BaseModel):
    explanation: str
    correct: bool


class EvaluationScore(BaseModel):
    explanation: str
    correct: int


class EvaluationPartialScore(BaseModel):
    """Evaluation with partial score support for list-based answers."""

    explanation: str
    correct: float = PydanticField(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 and 1.0. For non-list answers, must be exactly 0.0 or 1.0. For list answers, represents the ratio of correct items.",
    )


class NumericExtraction(BaseModel):
    extracted_value: str | None = PydanticField(
        ...,
        description="The numeric value extracted from the answer in string form without additional commentary. Use null if no numeric value is present.",
    )


class SequentialAnswer(BaseModel):
    scratchpad: str
    answer: str
    found_answer: bool


class WorkerResponse(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Worker's reasoning about the relevance and content of this chunk.",
    )
    evidence: str = PydanticField(
        ...,
        description='Key evidence found in this chunk relevant to the question, or "none" if nothing relevant.',
    )
    communication: str = PydanticField(
        ...,
        description="Updated accumulated summary of all evidence found so far (combining previous evidence with new findings from this chunk). This will be passed to the next worker.",
    )


class ManagerResponse(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Manager's reasoning about the synthesized evidence and how it answers the question.",
    )
    answer: str = PydanticField(
        ...,
        description="Final answer to the question based on the accumulated evidence.",
    )


class ChunkAnswer(BaseModel):
    answer: str
    found_answer: bool


class Action(BaseModel):
    reasoning: str
    run_sql: bool
    answer: str | None
    sql_query: str | None


class SQLAnswer(BaseModel):
    reasoning: str
    sql_query: str


class Normalization(BaseModel):
    currency: Optional[str] = None
    percent: Optional[str] = None
    date_format: Optional[str] = None


class Field(BaseModel):
    name: str
    data_type: str
    enum_values: Optional[list[str]] = None
    unit: Optional[str]
    scale: Optional[str]
    description: str
    required: bool
    normalization: Optional[Normalization] = None


class Table(BaseModel):
    name: str
    description: str
    fields: list[Field]


class Tables(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Reasong about which fields are required for extracting information from document chunks so that the question can be answered by combining the information from the chunks.",
    )
    tables: list[Table] = PydanticField(
        ...,
        description="The tables that are required for extracting information from document chunks so that the question can be answered by combining the information from the chunks.",
    )


class SchemaFeedbackUpdate(BaseModel):
    table_name: str = PydanticField(
        ...,
        description="Exact name of the table being updated.",
    )
    field_name: Optional[str] = PydanticField(
        default=None,
        description="Optional field name. Set to null when updating only the table description.",
    )
    updated_description: str = PydanticField(
        ...,
        description="Replacement description text capturing the clarified guidance.",
    )
    reasoning: str = PydanticField(
        ...,
        description="Short explanation of why this update fixes the mismatch.",
    )


class SchemaFeedbackResponse(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="High-level reasoning for the proposed schema description revisions.",
    )
    updates: list[SchemaFeedbackUpdate] = PydanticField(
        default_factory=list,
        description="List of focused updates to apply to the schema.",
    )


class NewFieldValues(BaseModel):
    row_number: int
    name: str
    reasoning: str
    value: str | float | int | bool | None
    field: str


class NewField(BaseModel):
    name: str
    description: str
    extraction_guideline: str
    data_type: str
    unit: str
    scale: str


class NewFields(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for which new fields are required. Each new field should be atomic and not be a combination of other fields.",
    )
    fields: list[NewField]
    # values: list[NewFieldValues]


class Column(BaseModel):
    reason: str = PydanticField(description="The reason for the decision to compute the new value for the field.")
    field_name: str = PydanticField(description="The name of the field that is being computed.")
    row_ids: list[int] = PydanticField(
        description="The row ids of the rows that are used to compute the new value for the field."
    )
    new_column_name: str = PydanticField(
        description="After the SQL query is executed, if there is a new column name for the exisiting field, then this should be the name of the new column."
    )


class Decision(BaseModel):
    reasoning: str = PydanticField(description="The reasoning for the decision to compute the new value for fields.")
    fields: list[Column] = PydanticField(description="The fields that are being computed and the new values for them.")


class Output(BaseModel):
    decision: Decision = PydanticField(description="The decision to compute the new value for fields.")
    sql_query: str = PydanticField(description="The SQL query to compute the new value for fields.")


class ObjectiveNecessity(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision if the objective is necessary cleaning the data.",
    )
    required: bool = PydanticField(
        ...,
        description="Whether the objective is necessary for the answer.",
    )


class TableOperation(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for how you will construct the sql query to perform the table operation.",
    )
    sql_query: str = PydanticField(
        ...,
        description="The SQL query to perform the table operation.",
    )


class ProvenanceSQL(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for how you will construct the sql query to get the provenance of the new table data.",
    )
    sql_query: str = PydanticField(
        ...,
        description="The SQL query to get the provenance of the new table data.",
    )


def create_dynamic_extraction_relation_model(
    tables: list[Table],
) -> Type[BaseModel]:
    relation_models = []
    for table in tables:
        field_models = {}
        for field in table.fields:
            # this data_type is different since we want to allow lists of types
            data_type = string_to_type(field.data_type)

            # get the list of types
            origin = get_origin(data_type)
            args = get_args(data_type)
            if origin is list and args:
                value_type = args[0]
            else:
                value_type = data_type

            field_models[field.name] = create_model(
                "Extracted",
                reasoning=(
                    str,
                    PydanticField(
                        ...,
                        description=(
                            "Short explanation of how this field's value was located and normalized for this row."
                        ),
                    ),
                ),
                value=(
                    Optional[value_type],
                    PydanticField(
                        ...,
                        description="Normalized value for this field, coerced to the schema's target type and unit.",
                    ),
                ),
                quote=(
                    Optional[list[str]],
                    PydanticField(
                        ...,
                        description=(
                            "Largest contiguous supporting text span(s) from the page for this value, or null if "
                            "no direct evidence is present."
                        ),
                    ),
                ),
                is_explicit=(
                    bool,
                    PydanticField(
                        ...,
                        description=(
                            "True if the value is explicitly stated in the text (including trivial normalization); "
                            "False if it required non-trivial inference or arithmetic."
                        ),
                    ),
                ),
                confidence=(
                    str,
                    PydanticField(
                        ...,
                        description=(
                            "Extractor confidence level for this field on this row; one of "
                            '"Very High", "High", "Medium", "Low", "Very Low".'
                        ),
                    ),
                ),
            )

        relation_models.append(create_model(table.name, **field_models))

    model = create_model(
        "ExtractionOutput",
        extraction_plan=(str, ...),
        **{model_name.__name__: (list[model_name], ...) for model_name in relation_models},
    )
    logger.info(pydantic_model_to_signature(model, func_name="ExtractionOutput"))
    return model


class DocumentDescriptions(BaseModel):
    descriptions: list[str] = PydanticField(
        ...,
        description="List of document descriptions, one per document.",
        min_items=1,
    )

    def __iter__(self):
        return iter(self.descriptions)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx]


class DocumentTitle(BaseModel):
    thought: str = PydanticField(
        ...,
        description="The thought process for the decision to extract/generate the title for the document.",
    )
    title: str = PydanticField(
        ...,
        description="The title of the document.",
    )


class ErrorAnalysisResponse(BaseModel):
    """Model for error analysis of a question's execution."""

    within_reason: bool = PydanticField(
        ...,
        description="Regardless if the actual evaluator marked this as correct or not, is the difference between the gold answers and the predicted answer within reason? Check if the predicted answer correctly captures the meaning of the gold answer, even if it provides more or fewer details, uses different wording, or varies in specificity.",
    )
    error_type: str = PydanticField(
        ...,
        description="The high-level categorization of the error (e.g., 'Schema Mismatch', 'Data Quality Issue', 'Reasoning Error')",
    )
    pipeline_stage: str = PydanticField(
        ...,
        description="The stage of the pipeline where the error occurred. If there are no relevant logs included in the input, then there is no way of knowing which stage the error was introduced in. The pipeline stage should thus be 'unknown'.",
        # Limit to known pipeline stages
        pattern="^(schema generation|extraction|merge|answer generation|unknown)$",
    )
    error_description: str = PydanticField(..., description="Detailed explanation of what went wrong and why")
    improvement_suggestion: str = PydanticField(
        ..., description="Concrete suggestions for how to fix or prevent this error in the future"
    )


class RephrasedQuestion(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Thought process for how to rephrase the question.",
    )
    question: str = PydanticField(
        ...,
        description="The rephrased question. The question should be different from the original question, but should be equivalent in meaning.",
    )


class ExtractionFallbackQuestion(BaseModel):
    """Response for generating a fallback extraction question when initial extraction yields no data."""

    fallback_question: str = PydanticField(
        ...,
        description="A rephrased question that focuses on extracting any available information for the schema fields from this document",
    )
    target_fields: list[str] = PydanticField(
        default_factory=list,
        description="List of field names the fallback question is specifically targeting",
    )
    reasoning: str = PydanticField(
        default="",
        description="Reasoning for why this question might extract more data from the document",
    )


class TableProcessingNeeded(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision to process the tables for this question. If the question can be directly answered from the tables, then we do not need to process the tables.",
    )
    processing_needed: bool = PydanticField(
        ...,
        description="Whether the tables should be processed for this question.",
    )


# ---------------------------------------------------------------------------
# Dynamic schema extraction models
# ---------------------------------------------------------------------------


class SchemaSufficiencyCheck(BaseModel):
    """LLM output for evaluating whether the current schema DAG covers a chunk."""

    reasoning: str = PydanticField(
        ...,
        description="Step-by-step reasoning about whether the existing schemas can capture all relevant information in this chunk.",
    )
    is_sufficient: bool = PydanticField(
        ...,
        description="True if the current schema DAG can represent all relevant information in the chunk.",
    )
    missing_information_summary: Optional[str] = PydanticField(
        default=None,
        description="If is_sufficient is False, a concise description of what entities, relationships, or attributes are missing from the current schemas.",
    )


class ProposedSchema(BaseModel):
    """A single new normalised table proposed during dynamic schema extension."""

    table_name: str = PydanticField(..., description="Machine-friendly name for the new table.")
    description: str = PydanticField(..., description="What relationship / entity this table captures.")
    fields: list[Field] = PydanticField(..., description="Field definitions for the new table.")
    foreign_key_table: str = PydanticField(
        ..., description="Name of the existing table in the DAG that this table links to."
    )
    foreign_key_fields: list[str] = PydanticField(
        ..., description="Field names in this new table that form the foreign key."
    )
    foreign_key_references: list[str] = PydanticField(
        ..., description="Field names in the referenced (parent) table that form that table's primary key."
    )


class SchemaExtensionProposal(BaseModel):
    """LLM output proposing one or more new schemas to add to the DAG."""

    reasoning: str = PydanticField(
        ...,
        description="Reasoning about what new schemas are needed and why they cannot be represented by existing ones.",
    )
    new_schemas: list[ProposedSchema] = PydanticField(
        ...,
        description="List of new normalised table schemas to add. Each must define at least one FK to an existing table in the DAG. Do NOT define primary keys.",
    )


class DefaultFieldValue(BaseModel):
    """A single field default value for an absent-table placeholder row."""

    field_name: str = PydanticField(..., description="Name of the field.")
    default_value: Optional[str] = PydanticField(
        default=None,
        description="Default value as a string (e.g. '0', 'None', 'N/A'). null if no default.",
    )


class AbsentTableDecision(BaseModel):
    """LLM decision for a single table absent from a document."""

    table_name: str = PydanticField(..., description="Name of the absent table.")
    action: Literal["skip", "add_default_row"] = PydanticField(
        ...,
        description=(
            "'skip' if absence means not investigated/not applicable; "
            "'add_default_row' if absence implies a meaningful default (e.g. zero)."
        ),
    )
    reasoning: str = PydanticField(..., description="Why this action was chosen for the table.")
    default_field_values: Optional[list[DefaultFieldValue]] = PydanticField(
        default=None,
        description="Per-field default values. Required when action is 'add_default_row'.",
    )

    @property
    def default_values(self) -> dict[str, Any] | None:
        """Convert list-based field values to dict for downstream consumption."""
        if self.default_field_values is None:
            return None
        return {fv.field_name: fv.default_value for fv in self.default_field_values}


class AbsenceArbiterOutput(BaseModel):
    """LLM output deciding how to handle tables absent from a document."""

    decisions: list[AbsentTableDecision] = PydanticField(..., description="One decision per absent table.")


@dataclass
class ExtractedTable:
    name: str
    tables: list[Table]
    sql_query: Output
    dataframe: pd.DataFrame
    dataframe_table_name: str
    table_str: str = None
    actions: list[dict] = None

    def to_template_dict(self) -> dict:
        """Convert to a simple dict for use in Jinja2 templates (which restrict attribute access)."""
        return {
            "name": self.name,
            "dataframe_table_name": self.dataframe_table_name,
            "table_str": self.table_str,
        }


# ---------------------------------------------------------------------------
# Selective extraction V2: Chunk Router models
# ---------------------------------------------------------------------------


class TableSelection(BaseModel):
    """A single table chosen by the Chunk Router for extraction."""

    table_name: str = PydanticField(..., description="Name of the existing table to extract into.")
    reason: str = PydanticField(..., description="Brief reason why this table is relevant to the chunk.")


class NewTableSuggestion(BaseModel):
    """A new table suggested by the Chunk Router."""

    table_name: str = PydanticField(..., description="Proposed name for the new table.")
    description: str = PydanticField(..., description="What this table should capture.")
    attach_to_table: str = PydanticField(
        ...,
        description="Name of an existing table this should FK to.",
    )
    field_sketches: list[str] = PydanticField(..., description="Brief list of field names the table should contain.")
    reasoning: str = PydanticField(
        default="",
        description=(
            "Why this table is needed: what concrete rows from the chunk would "
            "populate it, and why the FK to attach_to_table is fillable."
        ),
    )


class ChunkRoutingDecision(BaseModel):
    """Output of the Chunk Router: which tables to extract + new table ideas."""

    reasoning: str = PydanticField(
        ...,
        description="Step-by-step reasoning about what information the chunk contains and which tables are relevant.",
    )
    selected_tables: list[TableSelection] = PydanticField(
        ...,
        description="Existing tables to extract into for this chunk.",
    )
    new_table_suggestions: list[NewTableSuggestion] = PydanticField(
        default_factory=list,
        description="Suggestions for new tables not yet in the DAG. Empty if the schema is sufficient.",
    )
