import json
from pathlib import Path
import time
from typing import Any


from sliders.document import Document
from sliders.llm_models import Field, Tables, Table

from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from sliders.log_utils import logger

CURRENT_DIR = Path(__file__).parent


_REQUIRED_FIELD_KEYS = {"name", "data_type", "description", "required", "unit", "scale"}


def _field_is_complete(field: dict) -> bool:
    """Return True if a user-supplied field dict already has every required attribute."""
    return _REQUIRED_FIELD_KEYS.issubset(field.keys())


def _normalize_user_schema(user_schema: Any) -> dict:
    """Convert a loose user schema spec into the shape that `Tables(**...)` expects.

    Accepts any of the following:
      - a list of table dicts: ``[{"name": "T", "fields": ["a", "b"]}, ...]``
      - a dict with ``tables`` key: ``{"tables": [...]}``
      - field specs as plain strings (just the name) or dicts (partial metadata)

    Unknown keys on fields are preserved so the LLM completion step can see them.
    """
    if isinstance(user_schema, Tables):
        return user_schema.model_dump()

    if isinstance(user_schema, list):
        tables_list = user_schema
    elif isinstance(user_schema, dict) and "tables" in user_schema:
        tables_list = user_schema["tables"]
    else:
        raise ValueError(
            "user_schema must be a list of tables or a dict with a 'tables' key. "
            f"Got: {type(user_schema).__name__}"
        )

    normalized_tables = []
    for table in tables_list:
        if not isinstance(table, dict) or "name" not in table:
            raise ValueError(f"Each table must be a dict with a 'name'. Got: {table!r}")

        raw_fields = table.get("fields", [])
        normalized_fields = []
        for f in raw_fields:
            if isinstance(f, str):
                normalized_fields.append({"name": f})
            elif isinstance(f, dict) and "name" in f:
                normalized_fields.append(dict(f))
            else:
                raise ValueError(
                    f"Each field must be a string name or a dict with a 'name'. Got: {f!r}"
                )

        normalized_tables.append(
            {
                "name": table["name"],
                "description": table.get("description"),
                "fields": normalized_fields,
            }
        )

    return {
        "reasoning": user_schema.get("reasoning") if isinstance(user_schema, dict) else None,
        "tables": normalized_tables,
    }


def _user_schema_is_already_complete(normalized: dict) -> bool:
    """True if every field already has all attributes required by the Field pydantic model."""
    for table in normalized["tables"]:
        if not table.get("description"):
            return False
        for field in table["fields"]:
            if not _field_is_complete(field):
                return False
    return True


def _build_tables_from_normalized(normalized: dict) -> Tables:
    """Construct a valid ``Tables`` from a fully-specified normalized dict."""
    tables = []
    for t in normalized["tables"]:
        fields = []
        for f in t["fields"]:
            fields.append(
                Field(
                    name=f["name"],
                    data_type=f["data_type"],
                    enum_values=f.get("enum_values"),
                    unit=f.get("unit"),
                    scale=f.get("scale"),
                    description=f["description"],
                    required=f["required"],
                    normalization=f.get("normalization"),
                )
            )
        tables.append(Table(name=t["name"], description=t["description"], fields=fields))
    return Tables(
        reasoning=normalized.get("reasoning") or "User-provided schema.",
        tables=tables,
    )


class GenerateSchema:
    def __init__(self, config: dict, model_config: dict):
        self.config = config
        self.model_config = model_config
        self.library_of_guidelines = None

        if (
            self.config.get("library_of_guidelines_path", None)
            and self.config.get("generate_schema_type", "single_shot") == "library_based"
        ):
            self.library_of_guidelines = self.load_library_of_guidelines()

    def load_library_of_guidelines(self) -> dict:
        with open(Path(CURRENT_DIR) / "../" / self.config["library_of_guidelines_path"], "r") as f:
            return json.load(f)

    @staticmethod
    def create_generate_schema_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        generate_schema_template = load_fewshot_prompt_template(
            template_file="sliders/generate_schema_qa.prompt",
            template_blocks=[],
        )
        generate_schema_chain = generate_schema_template | llm_client.with_structured_output(Tables)
        return generate_schema_chain

    async def single_shot_generate(
        self, question: str, documents: list[Document], metadata: dict, task_guidelines: str | None
    ) -> Tables:
        schema_start_time = time.time()
        handler = LoggingHandler(
            prompt_file="sliders/generate_schema_qa.prompt",
            metadata={
                "question": question,
                "stage": "generate_schema",
                **(metadata or {}),
            },
        )
        try:
            schema = await self.create_generate_schema_chain(
                **self.model_config["generate_schema"],
            ).ainvoke(
                {
                    "document_description": documents[0].description,
                    "document_text": documents[0].content[:1000] + "..."
                    if self.config.get("add_document_text", False)
                    else None,
                    "question": question,
                    "guidelines": task_guidelines,
                    "enable_extraction_guidelines": self.config.get("enable_extraction_guidelines", False),
                },
                config={"callbacks": [handler]},
            )

            # Update schema metadata
            metadata["timing"]["schema_generation"]["generation_time"] = time.time() - schema_start_time
            metadata["schema"]["generated_classes"] = len(schema.tables)
            metadata["schema"]["total_fields"] = sum(len(cls.fields) for cls in schema.tables)
            metadata["schema"]["generation_time"] = time.time() - schema_start_time

            # Store schema complexity metrics
            metadata["schema"]["average_fields_per_class"] = (
                metadata["schema"]["total_fields"] / metadata["schema"]["generated_classes"]
                if metadata["schema"]["generated_classes"] > 0
                else 0
            )

            # Store field types distribution
            field_types = {}
            for cls in schema.tables:
                for field in cls.fields:
                    field_type = field.data_type
                    field_types[field_type] = field_types.get(field_type, 0) + 1
            metadata["schema"]["field_types_distribution"] = field_types

            # Store the schema object for backward compatibility
            metadata["schema"]["schema_object"] = schema.model_dump()

            if self.config.get("add_extra_information_class", False):
                schema.tables.append(
                    Table(
                        name="AdditionalInformation",
                        fields=[
                            Field(
                                name="additional_information",
                                data_type="str",
                                description="Additional information that is useful for answering the question, but isn't covered by the other relationship schema.",
                                unit=None,
                                scale=None,
                                required=False,
                                normalization=None,
                            )
                        ],
                    )
                )

            return schema

        except Exception as e:
            metadata["errors"].append(
                {
                    "stage": "schema_generation",
                    "error": str(e),
                    "question": question,
                    "document_descriptions": [doc.description for doc in documents],
                }
            )
            metadata["timing"]["schema_generation"]["generation_time"] = time.time() - schema_start_time
            raise

    async def library_based_generate(
        self,
        question: str,
        documents: list[Document],
        metadata: dict,
        guidelines: str | None,
        question_type: str | None = None,
        document_type: str | None = None,
    ) -> Tables:
        """
        Generate schema using library-based guidelines.
        Expects question_type and document_type to be provided (should be classified in system.py).
        Falls back to defaults if not provided.
        """
        start_time = time.time()

        # Use defaults if not provided
        if question_type is None:
            question_type = "simple"
        if document_type is None:
            document_type = "others"

        if self.library_of_guidelines["question_type"][question_type]["guidelines"] is not None:
            question_guidelines = "\n- ".join(self.library_of_guidelines["question_type"][question_type]["guidelines"])
        else:
            question_guidelines = ""
        if self.library_of_guidelines["document_type"][document_type]["guidelines"] is not None:
            document_guidelines = "\n- ".join(self.library_of_guidelines["document_type"][document_type]["guidelines"])
        else:
            document_guidelines = ""
        guidelines = question_guidelines + document_guidelines

        schema = await self.single_shot_generate(question, documents, metadata, guidelines)
        metadata["timing"]["schema_generation"]["library_based_selection_time"] = time.time() - start_time
        metadata["schema"]["generation_time"] = time.time() - start_time
        return schema

    async def generate_from_user_schema(
        self,
        user_schema: Any,
        question: str,
        documents: list[Document],
        metadata: dict,
    ) -> Tables:
        """Build a ``Tables`` object from a partial user-provided schema.

        Any field missing pydantic-required metadata (``data_type``,
        ``description``, ``required``, ``unit``, ``scale``) is completed by a
        single LLM call using ``sliders/complete_user_schema.prompt``. The
        LLM is instructed to preserve everything the user explicitly provided
        and to NOT add new tables or fields.
        """
        start_time = time.time()
        normalized = _normalize_user_schema(user_schema)

        if _user_schema_is_already_complete(normalized):
            logger.info("User schema is fully specified; skipping LLM completion.")
            schema = _build_tables_from_normalized(normalized)
        else:
            logger.info("Completing partial user schema via LLM...")
            handler = LoggingHandler(
                prompt_file="sliders/complete_user_schema.prompt",
                metadata={
                    "question": question,
                    "stage": "complete_user_schema",
                    **(metadata or {}),
                },
            )
            chain_kwargs = dict(self.model_config["generate_schema"])
            llm_client = get_llm_client(**chain_kwargs)
            template = load_fewshot_prompt_template(
                template_file="sliders/complete_user_schema.prompt",
                template_blocks=[],
            )
            chain = template | llm_client.with_structured_output(Tables)
            schema = await chain.ainvoke(
                {
                    "question": question,
                    "document_description": documents[0].description if documents else "",
                    "user_schema": json.dumps(normalized, indent=2),
                },
                config={"callbacks": [handler]},
            )

        metadata["timing"]["schema_generation"]["generation_time"] = time.time() - start_time
        metadata["schema"]["generated_classes"] = len(schema.tables)
        metadata["schema"]["total_fields"] = sum(len(t.fields) for t in schema.tables)
        metadata["schema"]["generation_time"] = time.time() - start_time
        metadata["schema"]["average_fields_per_class"] = (
            metadata["schema"]["total_fields"] / metadata["schema"]["generated_classes"]
            if metadata["schema"]["generated_classes"] > 0
            else 0
        )
        metadata["schema"]["schema_object"] = schema.model_dump()
        metadata["schema"]["user_provided"] = True
        return schema

    async def generate(
        self,
        question: str,
        documents: list[Document],
        metadata: dict,
        task_guidelines: str | None,
        question_type: str | None = None,
        document_type: str | None = None,
    ) -> Tables:
        user_schema = self.config.get("user_schema")
        if user_schema is not None:
            return await self.generate_from_user_schema(user_schema, question, documents, metadata)

        if self.config.get("generate_schema_type", "single_shot") == "single_shot":
            return await self.single_shot_generate(question, documents, metadata, task_guidelines)
        elif self.config.get("generate_schema_type", "single_shot") == "library_based":
            return await self.library_based_generate(
                question, documents, metadata, task_guidelines, question_type=question_type, document_type=document_type
            )
        else:
            raise ValueError(f"Invalid generate schema type: {self.config.get('generate_schema_type', 'single_shot')}")
