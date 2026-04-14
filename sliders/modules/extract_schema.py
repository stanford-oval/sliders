from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Sequence, Literal

from pydantic import BaseModel, ConfigDict, Field

from sliders.callbacks.logging import LoggingHandler
from sliders.document import Document
from sliders.globals import SlidersGlobal
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.llm_models import (
    AbsenceArbiterOutput,
    AbsentTableDecision,
    ExtractionFallbackQuestion,
    InformationDensityResponse,
    IsRelevantPage,
    RowExtractionInstructions,
    Tables,
)
from sliders.log_utils import logger
from sliders.utils import prepare_schema_repr


class FieldExtraction(BaseModel):
    """Normalized representation of a single extracted field."""

    model_config = ConfigDict(extra="allow")

    value: Any | None = None
    quote: list[str] | str | None = None
    rationale: str | None = None
    is_explicit: bool | None = None
    confidence: Literal["Very High", "High", "Medium", "Low", "Very Low"] | None = None


class RowExtraction(BaseModel):
    """Normalized representation of a row with metadata and named fields."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    metadata: dict[str, Any] = Field(default_factory=dict, alias="__metadata__")
    fields: dict[str, FieldExtraction] = Field(default_factory=dict)

    def to_json(self, include_quotes: bool) -> dict[str, Any]:
        """Dump to JSON-ready dict, optionally removing quote text."""
        serialized_fields = {}
        for name, payload in self.fields.items():
            payload_copy = payload.model_copy()
            if not include_quotes:
                payload_copy.quote = None
            serialized_fields[name] = payload_copy.model_dump(exclude_none=True)
        return {"__metadata__": self.metadata, "fields": serialized_fields}


class ExtractSchema:
    """Schema extraction orchestrator with safer JSON normalization."""

    def __init__(self, config: dict, model_config: dict):
        self.config = config
        self.model_config = model_config
        self.extract_quotes = self.config.get("extract_quotes", True)
        self.enable_verbalization_instructions = self.config.get("enable_verbalization_instructions", True)
        self._semaphore = asyncio.Semaphore(self.config.get("max_concurrent_calls", 16))
        self._arbiter_chain = None

    @staticmethod
    def _format_extraction_guidelines(schema: Tables) -> str | None:
        guidelines = []
        for table in schema.tables:
            for field in table.fields:
                if field.extraction_guideline:
                    guidelines.append(f"- **{field.name}**: {field.extraction_guideline}")
        return "\n".join(guidelines) if guidelines else None

    @staticmethod
    def create_extract_schema_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        extract_schema_template = load_fewshot_prompt_template(
            template_file="sliders/extract_schema.prompt",
            template_blocks=[],
        )
        return extract_schema_template | llm_client.with_structured_output(method="json_mode")

    @staticmethod
    def create_is_relevant_chunk_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        is_relevant_chunk_template = load_fewshot_prompt_template(
            template_file="sliders/is_relevant_chunk.prompt",
            template_blocks=[],
        )
        return is_relevant_chunk_template | llm_client.with_structured_output(IsRelevantPage)

    @staticmethod
    def create_info_density_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        info_density_template = load_fewshot_prompt_template(
            template_file="sliders/check_information_density.prompt",
            template_blocks=[],
        )
        return info_density_template | llm_client.with_structured_output(InformationDensityResponse)

    @staticmethod
    def create_row_instructions_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        row_instructions_template = load_fewshot_prompt_template(
            template_file="sliders/generate_row_instructions.prompt",
            template_blocks=[],
        )
        return row_instructions_template | llm_client.with_structured_output(RowExtractionInstructions)

    @staticmethod
    def create_fallback_question_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        fallback_question_template = load_fewshot_prompt_template(
            template_file="sliders/extraction_fallback_question.prompt",
            template_blocks=[],
        )
        return fallback_question_template | llm_client.with_structured_output(ExtractionFallbackQuestion)

    async def _invoke_with_limit(self, chain, payload: dict, handler: LoggingHandler | None = None):
        callbacks = [handler] if handler else []
        async with self._semaphore:
            return await chain.ainvoke(payload, config={"callbacks": callbacks})

    def _document_chunk_repr(self, document: Document, chunk: dict, page_number: int) -> str:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        return f"""
Headers: {metadata.get("headers", "")}
Page Number: {page_number}
Content:
{chunk.get("content", "") if isinstance(chunk, dict) else ""}"""

    def _coerce_result_to_dict(self, res: Any) -> dict | None:
        if res is None:
            return None
        if isinstance(res, str):
            try:
                return json.loads(res)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON extraction payload; skipping")
                return None
        if hasattr(res, "model_dump"):
            return res.model_dump()
        if hasattr(res, "dict"):
            return res.dict()
        if isinstance(res, dict):
            return res
        logger.warning(f"Unsupported extraction result type {type(res)}; skipping")
        return None

    def _normalize_field_payload(self, payload: Any) -> FieldExtraction:
        if isinstance(payload, FieldExtraction):
            return payload
        if payload is None:
            return FieldExtraction()
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        if isinstance(payload, dict):
            return FieldExtraction(**payload)
        return FieldExtraction(value=payload)

    def _normalize_row_payload(self, row_payload: Any, base_metadata: dict[str, Any]) -> RowExtraction | None:
        if isinstance(row_payload, RowExtraction):
            merged_metadata = base_metadata | row_payload.metadata
            return RowExtraction(metadata=merged_metadata, fields=row_payload.fields)
        if row_payload is None:
            return None
        if hasattr(row_payload, "model_dump"):
            row_payload = row_payload.model_dump()
        if not isinstance(row_payload, dict):
            return None

        merged_metadata = base_metadata | row_payload.get("__metadata__", {})
        fields: dict[str, FieldExtraction] = {}
        for field_name, field_payload in row_payload["fields"].items():
            if field_name == "__metadata__":
                continue
            fields[field_name] = self._normalize_field_payload(field_payload)
        return RowExtraction(metadata=merged_metadata, fields=fields)

    def _normalize_relationship_rows(self, values: Any, base_metadata: dict[str, Any]) -> list[RowExtraction]:
        if values is None:
            return []
        rows_iter: Iterable[Any]
        if isinstance(values, dict):
            rows_iter = [values]
        elif isinstance(values, (list, tuple, set)):
            rows_iter = values
        else:
            logger.warning(f"Unexpected row container type {type(values)}; skipping")
            return []

        normalized_rows: list[RowExtraction] = []
        for row_payload in rows_iter:
            row = self._normalize_row_payload(row_payload, base_metadata)
            if row:
                normalized_rows.append(row)
        return normalized_rows

    def _merge_chunk_results(
        self, results: Sequence[Any], base_metadata: dict[str, Any]
    ) -> dict[str, list[RowExtraction]]:
        merged: dict[str, list[RowExtraction]] = defaultdict(list)
        for res in results:
            chunk = self._coerce_result_to_dict(res)
            if not chunk:
                continue
            if "verbalization" in chunk and isinstance(chunk["verbalization"], str):
                logger.info(f"VERBALIZATION: {chunk['verbalization']}")
            for relationship_name, rows in chunk.items():
                if relationship_name in {"extraction_plan", "verbalization"}:
                    continue
                merged[relationship_name].extend(self._normalize_relationship_rows(rows, base_metadata))
        return {name: rows for name, rows in merged.items() if rows}

    def convert_extracted_data_to_json(self, extracted_data: list[Any], document: Document, doc_id: int) -> list[dict]:
        final_json_repr = []
        for chunk_id, raw in enumerate(extracted_data):
            base_metadata = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_name": document.document_name,
                "chunk_header": document.chunks[chunk_id].get("metadata", {}),
            }
            chunk = self._coerce_result_to_dict(raw)
            if not chunk:
                continue

            # Handle schema-guided tables payloads of the form:
            # {"tables": [{"name": "...", "rows": [...], "__metadata__": {...}}, ...]}
            try:
                for table_obj in chunk["tables"]:
                    table_name = table_obj.get("name")
                    table_rows = table_obj.get("rows", [])
                    table_metadata = table_obj.get("__metadata__", {})
                    table_base_metadata = base_metadata | table_metadata
                    normalized_rows = self._normalize_relationship_rows(table_rows, table_base_metadata)
                    if normalized_rows:
                        final_json_repr.append(
                            {table_name: [row.to_json(include_quotes=self.extract_quotes) for row in normalized_rows]}
                        )
            except Exception as e:
                logger.error(f"Error converting extracted data to JSON: {e}")
                continue

            if "verbalization" in chunk and isinstance(chunk["verbalization"], str):
                logger.info(f"VERBALIZATION (Chunk {chunk_id}, Doc={document.document_name}): {chunk['verbalization']}")

        return final_json_repr

    async def _extract_chunk_single(
        self,
        question_id: str,
        extract_chain,
        schema_repr: str,
        extraction_guidelines_repr: str | None,
        document: Document,
        chunk: dict,
        chunk_id: int,
        question: str,
        task_guidelines: str,
    ):
        if self.config.get("is_relevant_chunk", False):
            is_relevant_handler = LoggingHandler(
                prompt_file="sliders/is_relevant_chunk.prompt",
                metadata={
                    "question_id": question_id,
                    "question": question,
                    "stage": "is_relevant_chunk",
                    "chunk_id": chunk_id,
                    "document_name": document.document_name,
                    "objective": str(chunk_id) + "_" + document.document_name,
                },
            )
            is_relevant_chain = self.create_is_relevant_chunk_chain(**self.model_config["is_relevant_chunk"])
            is_relevant = await self._invoke_with_limit(
                is_relevant_chain,
                {
                    "document": self._document_chunk_repr(document, chunk, chunk_id),
                    "question": question,
                },
                handler=is_relevant_handler,
            )
            if not is_relevant.is_relevant:
                return "irrelevant"

        if self.config.get("check_info_density", False):
            density_handler = LoggingHandler(
                prompt_file="sliders/check_information_density.prompt",
                metadata={
                    "question_id": question_id,
                    "question": question,
                    "stage": "check_info_density",
                    "chunk_id": chunk_id,
                    "document_name": document.document_name,
                    "objective": str(chunk_id) + "_density_" + document.document_name,
                },
            )
            density_chain = self.create_info_density_chain(**self.model_config.get("check_info_density", {}))
            density_response: InformationDensityResponse = await self._invoke_with_limit(
                density_chain,
                {
                    "document": self._document_chunk_repr(document, chunk, chunk_id),
                    "question": question,
                    "schema": schema_repr,
                },
                handler=density_handler,
            )
            logger.info(
                f"Density check doc={document.document_name} chunk={chunk_id}: "
                f"estimated_rows={density_response.estimated_row_count} is_dense={density_response.is_dense}"
            )
            density_threshold = self.config.get("info_density_threshold", 3)
            if density_response.is_dense and density_response.estimated_row_count >= density_threshold:
                row_instructions_handler = LoggingHandler(
                    prompt_file="sliders/generate_row_instructions.prompt",
                    metadata={
                        "question_id": question_id,
                        "question": question,
                        "stage": "generate_row_instructions",
                        "chunk_id": chunk_id,
                        "document_name": document.document_name,
                        "objective": f"{chunk_id}_row_instructions_{document.document_name}",
                    },
                )
                row_instructions_chain = self.create_row_instructions_chain(
                    **self.model_config.get("generate_row_instructions", {})
                )
                row_instructions_response: RowExtractionInstructions = await self._invoke_with_limit(
                    row_instructions_chain,
                    {
                        "document": self._document_chunk_repr(document, chunk, chunk_id),
                        "question": question,
                        "schema": schema_repr,
                        "estimated_row_count": density_response.estimated_row_count,
                    },
                    handler=row_instructions_handler,
                )
                return await self._extract_chunk_dense(
                    question_id=question_id,
                    extract_chain=extract_chain,
                    schema_repr=schema_repr,
                    extraction_guidelines_repr=extraction_guidelines_repr,
                    document=document,
                    chunk=chunk,
                    chunk_id=chunk_id,
                    question=question,
                    task_guidelines=task_guidelines,
                    row_instructions=row_instructions_response.row_descriptions,
                )

        handler = LoggingHandler(
            prompt_file="sliders/extract_schema.prompt",
            metadata={
                "question_id": question_id,
                "question": question,
                "stage": "extract_chunk_single",
                "chunk_id": chunk_id,
                "document_name": document.document_name,
                "objective": str(chunk_id) + "_" + document.document_name,
            },
        )
        prev_summary = ""
        if self.config.get("use_previous_chunk_summary", False):
            prev_summary_key = self.config.get("previous_summary_key", "narrative_prev_summary")
            prev_summary = chunk.get("metadata", {}).get(prev_summary_key, "")

        payload = {
            "schema": schema_repr,
            "extraction_guidelines": extraction_guidelines_repr,
            "document": self._document_chunk_repr(document, chunk, chunk_id),
            "document_name": document.document_name,
            "document_description": document.description,
            "question": question,
            "task_guidelines": task_guidelines,
            "previous_chunk_summary": prev_summary,
            "extract_quotes": self.extract_quotes,
            "enable_verbalization_instructions": self.enable_verbalization_instructions,
        }
        return await self._invoke_with_limit(extract_chain, payload, handler=handler)

    async def _extract_chunk_dense(
        self,
        question_id: str,
        extract_chain,
        schema_repr: str,
        extraction_guidelines_repr: str | None,
        document: Document,
        chunk: dict,
        chunk_id: int,
        question: str,
        task_guidelines: str,
        row_instructions: list[str],
    ):
        logger.info(
            f"Row-by-row extraction doc={document.document_name} chunk={chunk_id} with {len(row_instructions)} instructions"
        )
        row_tasks = []
        base_question = question
        for row_idx, row_instruction in enumerate(row_instructions):
            focused_question = f"{base_question}\n\nFOCUS: {row_instruction}"
            handler = LoggingHandler(
                prompt_file="sliders/extract_schema.prompt",
                metadata={
                    "question_id": question_id,
                    "question": focused_question,
                    "stage": "extract_chunk_dense_row",
                    "chunk_id": chunk_id,
                    "row_index": row_idx,
                    "document_name": document.document_name,
                    "objective": f"{chunk_id}_{row_idx}_{document.document_name}",
                },
            )
            prev_summary = ""
            if self.config.get("use_previous_chunk_summary", False):
                prev_summary_key = self.config.get("previous_summary_key", "narrative_prev_summary")
                prev_summary = chunk.get("metadata", {}).get(prev_summary_key, "")
            payload = {
                "schema": schema_repr,
                "extraction_guidelines": extraction_guidelines_repr,
                "document": self._document_chunk_repr(document, chunk, chunk_id),
                "document_name": document.document_name,
                "document_description": document.description,
                "question": focused_question,
                "task_guidelines": task_guidelines,
                "previous_chunk_summary": prev_summary,
                "extract_quotes": self.extract_quotes,
                "enable_verbalization_instructions": self.enable_verbalization_instructions,
            }
            row_tasks.append(self._invoke_with_limit(extract_chain, payload, handler=handler))

        results = await asyncio.gather(*row_tasks, return_exceptions=True)
        filtered_results: list[Any] = []
        for row_idx, result in enumerate(results):
            if result is None or isinstance(result, Exception):
                logger.warning(
                    f"Failed dense row {row_idx} for doc={document.document_name} chunk={chunk_id}: "
                    f"{result if isinstance(result, Exception) else 'None result'}"
                )
                continue
            filtered_results.append(result)

        if not filtered_results:
            logger.warning(f"No valid dense rows for doc={document.document_name} chunk={chunk_id}")
            return None

        base_metadata = {
            "chunk_id": chunk_id,
            "document_name": document.document_name,
        }
        merged_result = self._merge_chunk_results(filtered_results, base_metadata=base_metadata)
        logger.info(
            f"Dense merged {len(filtered_results)} results into {len(merged_result)} relationships "
            f"for doc={document.document_name} chunk={chunk_id}"
        )
        return {rel: [row.model_dump(by_alias=True) for row in rows] for rel, rows in merged_result.items()}

    async def handle_failed_extractions(
        self,
        question_id: str,
        question: str,
        extracted_data: list[Any],
        metadata: dict,
        schema: Tables,
        document: Document,
        schema_repr: str,
        successful_extractions: int,
        failed_extractions: int,
        retry_attempts: int,
        irrelevant_chunk_indexes: set[int] | None = None,
    ) -> tuple[int, int, int]:
        updated_success = successful_extractions
        updated_failure = failed_extractions
        updated_retries = retry_attempts
        skip_indices = irrelevant_chunk_indexes or set()

        for i, res in enumerate(extracted_data):
            if i in skip_indices:
                continue
            if isinstance(res, Exception) or res is None:
                updated_retries += 1
                try:
                    handler = LoggingHandler(
                        prompt_file="sliders/extract_schema.prompt",
                        metadata={
                            "question_id": question_id,
                            "question": question,
                            "stage": "extraction_retry",
                            "chunk_index": i,
                            "document": document.document_name,
                        },
                    )
                    retry_payload = deepcopy(self.model_config["extract_schema"])
                    retry_payload.pop("temperature", None)
                    extraction_chain = self.create_extract_schema_chain(temperature=0.7, **retry_payload)
                    prev_summary = ""
                    if self.config.get("use_previous_chunk_summary", False):
                        prev_key = self.config.get("previous_summary_key", "narrative_prev_summary")
                        prev_summary = document.chunks[i].get("metadata", {}).get(prev_key, "")
                    extraction_guidelines_repr = (
                        self._format_extraction_guidelines(schema)
                        if self.config.get("enable_extraction_guidelines", False)
                        else None
                    )
                    new_res = await self._invoke_with_limit(
                        extraction_chain,
                        {
                            "document": self._document_chunk_repr(document, document.chunks[i], i),
                            "document_name": document.document_name,
                            "document_description": document.description,
                            "schema": schema_repr,
                            "question": question,
                            "previous_chunk_summary": prev_summary,
                            "extraction_guidelines": extraction_guidelines_repr,
                            "extract_quotes": self.extract_quotes,
                            "enable_verbalization_instructions": self.enable_verbalization_instructions,
                        },
                        handler=handler,
                    )
                    extracted_data[i] = new_res
                    if new_res is not None:
                        updated_success += 1
                    else:
                        updated_failure += 1
                except Exception as e:  # pragma: no cover - defensive
                    metadata["errors"].append(
                        {
                            "stage": "extraction_retry",
                            "error": str(e),
                            "chunk_index": i,
                            "document": document.document_name,
                        }
                    )
                    extracted_data[i] = None
                    updated_failure += 1
            else:
                updated_success += 1
        return updated_success, updated_failure, updated_retries

    def _get_arbiter_chain(self):
        if self._arbiter_chain is None:
            cfg = self.model_config.get(
                "absent_table_arbiter",
                self.model_config.get("extract_schema", {}),
            )
            llm = get_llm_client(**cfg)
            template = load_fewshot_prompt_template(
                template_file="sliders/absent_table_arbiter.prompt",
                template_blocks=[],
            )
            self._arbiter_chain = template | llm.with_structured_output(AbsenceArbiterOutput)
        return self._arbiter_chain

    async def _decide_absent_table_handling(
        self,
        per_document_extracted_data: list[list[dict]],
        documents: list[Document],
        schema: Tables,
        question: str,
    ) -> dict[str, dict[str, AbsentTableDecision]]:
        """Identify tables absent across entire documents and ask the arbiter LLM."""
        docs_with_rows: dict[str, set[str]] = defaultdict(set)
        for doc_idx, document_data in enumerate(per_document_extracted_data):
            doc_name = documents[doc_idx].document_name if doc_idx < len(documents) else f"doc_{doc_idx}"
            if not document_data:
                continue
            for chunk_data in document_data:
                if not chunk_data:
                    continue
                for table_name, rows in chunk_data.items():
                    if rows:
                        docs_with_rows[table_name].add(doc_name)

        all_doc_names = {doc.document_name for doc in documents}
        absent_tables_per_doc: dict[str, set[str]] = {}
        for table in schema.tables:
            missing = all_doc_names - docs_with_rows.get(table.name, set())
            for doc_name in missing:
                absent_tables_per_doc.setdefault(doc_name, set()).add(table.name)

        if not absent_tables_per_doc:
            return {}

        field_map = {t.name: t.fields for t in schema.tables}
        desc_map = {t.name: t.description for t in schema.tables}
        doc_desc_map = {
            d.document_name: getattr(d, "document_description", "No description available.") for d in documents
        }

        decisions: dict[str, dict[str, AbsentTableDecision]] = {}
        for doc_name, absent_tables in absent_tables_per_doc.items():
            if not absent_tables:
                continue

            absent_desc_parts: list[str] = []
            for tname in sorted(absent_tables):
                table_desc = desc_map.get(tname, "")
                fields = field_map.get(tname, [])
                fields_str = ", ".join(f.name for f in fields)
                absent_desc_parts.append(f"- **{tname}**: {table_desc}\n  Fields: {fields_str}")

            try:
                chain = self._get_arbiter_chain()
                arbiter_output: AbsenceArbiterOutput = await chain.ainvoke(
                    {
                        "question": question,
                        "document_name": doc_name,
                        "document_description": doc_desc_map.get(doc_name, ""),
                        "absent_tables_description": "\n".join(absent_desc_parts),
                    }
                )
                doc_decisions: dict[str, AbsentTableDecision] = {}
                for dec in arbiter_output.decisions:
                    doc_decisions[dec.table_name] = dec
                    logger.info(f"[AbsenceArbiter] {doc_name} / {dec.table_name}: {dec.action} — {dec.reasoning}")
                decisions[doc_name] = doc_decisions
            except Exception as exc:
                logger.warning(
                    f"[AbsenceArbiter] LLM call failed for doc '{doc_name}': {exc}. "
                    f"Defaulting to 'skip' for all {len(absent_tables)} absent table(s)."
                )
                decisions[doc_name] = {
                    tname: AbsentTableDecision(
                        table_name=tname,
                        action="skip",
                        reasoning=f"Arbiter LLM call failed: {exc}",
                    )
                    for tname in absent_tables
                }

        return decisions

    def finalize_tables(
        self,
        per_document_extracted_data: list[list[dict]],
        documents: list[Document] | None = None,
        schema: Tables | None = None,
        absent_decisions: dict[str, dict[str, AbsentTableDecision]] | None = None,
    ) -> dict[str, list[dict]]:
        tables: dict[str, list[dict]] = {}
        if absent_decisions is None:
            absent_decisions = {}
        docs_with_rows: dict[str, set[str]] = defaultdict(set)

        for doc_idx, document_data in enumerate(per_document_extracted_data):
            doc_name = documents[doc_idx].document_name if documents and doc_idx < len(documents) else f"doc_{doc_idx}"

            if not document_data:
                continue
            for chunk_data in document_data:
                if not chunk_data:
                    continue

                for relationship_name, rows in chunk_data.items():
                    tables.setdefault(relationship_name, []).extend(rows)
                    if rows:
                        docs_with_rows[relationship_name].add(doc_name)

        if documents and schema:
            all_doc_names = {doc.document_name for doc in documents}
            field_map = {t.name: t.fields for t in schema.tables}

            for table in schema.tables:
                table_name = table.name
                if table_name not in tables:
                    tables[table_name] = []

                docs_missing = all_doc_names - docs_with_rows.get(table_name, set())

                for doc_name in docs_missing:
                    doc_decisions = absent_decisions.get(doc_name, {})
                    decision = doc_decisions.get(table_name)

                    if decision and decision.action == "skip":
                        logger.info(f"Skipping placeholder for '{table_name}' in '{doc_name}': {decision.reasoning}")
                        continue

                    if decision and decision.action == "add_default_row":
                        default_vals = decision.default_values or {}
                        placeholder_fields = {}
                        for field in field_map.get(table_name, []):
                            val = default_vals.get(field.name)
                            placeholder_fields[field.name] = {
                                "value": val,
                                "quote": None,
                                "rationale": f"Domain default: {decision.reasoning}",
                                "is_explicit": False,
                            }
                        tables[table_name].append(
                            {
                                "__metadata__": {
                                    "chunk_id": None,
                                    "document_id": None,
                                    "document_name": doc_name,
                                    "chunk_header": {},
                                    "is_placeholder": True,
                                    "placeholder_reason": "domain_default",
                                },
                                "fields": placeholder_fields,
                            }
                        )
                        logger.info(
                            f"Added default-row placeholder for '{table_name}' in '{doc_name}': {decision.reasoning}"
                        )
                        continue

                    # No arbiter decision — default to skip (safe)
                    logger.info(f"No arbiter decision for '{table_name}' in '{doc_name}'; skipping placeholder.")

        return tables

    async def checkback_extracted_data(
        self,
        question_id: str,
        extracted_data: list[dict],
        document: Document,
        doc_id: int,
        schema: Tables,
        question: str,
    ):
        logger.info(f"checkback_extracted_data: {len(extracted_data)} chunks for document {document.document_name}")
        if not self.extract_quotes:
            return

        for extracted_chunk_data in extracted_data:
            if not extracted_chunk_data:
                continue
            for relationship_name, rows in list(extracted_chunk_data.items()):
                cleaned_rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for field_name, field_value in row.items():
                        if field_name == "__metadata__":
                            continue
                        if (
                            isinstance(field_value, dict)
                            and field_value.get("quote") is None
                            and field_value.get("is_explicit") is False
                        ):
                            field_value["value"] = None
                    cleaned_rows.append(row)
                extracted_chunk_data[relationship_name] = cleaned_rows

    async def generate_fallback_question(
        self,
        question: str,
        schema: Tables,
        document: Document,
        tables_needing_data: list[str],
        metadata: dict,
    ) -> ExtractionFallbackQuestion | None:
        """Generate a fallback question for a document that yielded no extraction data."""
        handler = LoggingHandler(
            prompt_file="sliders/extraction_fallback_question.prompt",
            metadata={
                "question_id": metadata.get("question_id"),
                "question": question,
                "stage": "generate_fallback_question",
                "document_name": document.document_name,
            },
        )

        # Reuse extract_schema model config
        fallback_chain = self.create_fallback_question_chain(**self.model_config["extract_schema"])

        # Get sample content from first chunk
        document_sample = ""
        if document.chunks:
            document_sample = document.chunks[0].get("content", "")[:2000]

        schema_repr = prepare_schema_repr(schema)
        tables_repr = "\n".join(f"- {t}" for t in tables_needing_data)

        try:
            result = await self._invoke_with_limit(
                fallback_chain,
                {
                    "question": question,
                    "schema": schema_repr,
                    "document_name": document.document_name,
                    "document_description": document.description or "No description available",
                    "document_sample": document_sample,
                    "tables_needing_data": tables_repr,
                },
                handler=handler,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to generate fallback question for {document.document_name}: {e}")
            return None

    async def run_fallback_extraction(
        self,
        original_question: str,
        schema: Tables,
        document: Document,
        doc_idx: int,
        tables_needing_data: list[str],
        metadata: dict,
        task_guidelines: str,
    ) -> list[dict] | None:
        """Run fallback extraction for a document that yielded no data."""

        # Generate fallback question
        fallback_result = await self.generate_fallback_question(
            question=original_question,
            schema=schema,
            document=document,
            tables_needing_data=tables_needing_data,
            metadata=metadata,
        )

        if fallback_result is None or not fallback_result.fallback_question.strip():
            logger.info(f"No fallback question generated for {document.document_name}")
            return None

        fallback_question = fallback_result.fallback_question
        logger.info(f"Fallback question for {document.document_name}: {fallback_question}")

        # Track fallback attempt in metadata
        if "fallback_extractions" not in metadata["extraction"]:
            metadata["extraction"]["fallback_extractions"] = []
        metadata["extraction"]["fallback_extractions"].append(
            {
                "document_name": document.document_name,
                "original_question": original_question,
                "fallback_question": fallback_question,
                "reasoning": fallback_result.reasoning,
                "target_fields": fallback_result.target_fields,
            }
        )

        # Re-run extraction with fallback question
        schema_repr = prepare_schema_repr(schema)
        extraction_guidelines_repr = (
            self._format_extraction_guidelines(schema)
            if self.config.get("enable_extraction_guidelines", False)
            else None
        )

        extract_payload = deepcopy(self.model_config["extract_schema"])
        extract_chain = self.create_extract_schema_chain(**extract_payload)

        # Extract from all chunks with the fallback question
        logger.info(f"Fallback: processing {len(document.chunks)} chunks for {document.document_name}")
        tasks = []
        for chunk_id, chunk in enumerate(document.chunks):
            tasks.append(
                self._extract_chunk_single(
                    question_id=metadata.get("question_id"),
                    extract_chain=extract_chain,
                    schema_repr=schema_repr,
                    extraction_guidelines_repr=extraction_guidelines_repr,
                    document=document,
                    chunk=chunk,
                    chunk_id=chunk_id,
                    question=fallback_question,
                    task_guidelines=task_guidelines,
                )
            )

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            extracted_data = []
            for chunk_id, result in enumerate(results):
                if result == "irrelevant" or isinstance(result, Exception) or result is None:
                    extracted_data.append(None)
                else:
                    extracted_data.append(result)

            # Convert to JSON format
            json_repr = self.convert_extracted_data_to_json(extracted_data, document, doc_idx)

            if json_repr:
                logger.info(f"Fallback extraction succeeded for {document.document_name}: got {len(json_repr)} results")
            else:
                logger.info(f"Fallback extraction yielded no results for {document.document_name}")

            return json_repr

        except Exception as e:
            logger.error(f"Fallback extraction failed for {document.document_name}: {e}")
            return None

    async def _run_extraction_fallback(
        self,
        question: str,
        schema: Tables,
        documents: list[Document],
        metadata: dict,
        task_guidelines: str,
        per_document_extracted_data: list[list[dict]],
        per_document_irrelevant_chunks: list[set[int]],
    ) -> list[list[dict]]:
        """
        Run fallback extraction for documents that yielded no data for any table.

        Returns updated per_document_extracted_data with fallback results merged in.
        """
        table_names = [t.name for t in schema.tables]

        for doc_idx, document in enumerate(documents):
            doc_data = per_document_extracted_data[doc_idx]
            irrelevant_chunks = per_document_irrelevant_chunks[doc_idx]

            # Check if ALL chunks were marked irrelevant
            total_chunks = len(document.chunks)
            all_chunks_irrelevant = len(irrelevant_chunks) == total_chunks

            if not all_chunks_irrelevant:
                # At least one chunk was relevant, skip fallback
                relevant_chunks_count = total_chunks - len(irrelevant_chunks)
                logger.info(
                    f"Document '{document.document_name}' had {relevant_chunks_count} "
                    f"relevant chunk(s) but yielded no data. Skipping fallback extraction."
                )
                continue

            # Check which tables have data for this document
            tables_with_data = set()
            for chunk_data in doc_data:
                if chunk_data:
                    tables_with_data.update(chunk_data.keys())

            # Find tables that need data
            tables_needing_data = [t for t in table_names if t not in tables_with_data]

            if not tables_needing_data:
                # Document has data for all tables, no fallback needed
                continue

            logger.info(
                f"Document '{document.document_name}' had ALL chunks marked irrelevant and "
                f"missing data for tables: {tables_needing_data}. Running fallback extraction..."
            )

            # Run fallback extraction
            fallback_results = await self.run_fallback_extraction(
                original_question=question,
                schema=schema,
                document=document,
                doc_idx=doc_idx,
                tables_needing_data=tables_needing_data,
                metadata=metadata,
                task_guidelines=task_guidelines,
            )

            if fallback_results:
                # Merge fallback results into the document's extracted data
                per_document_extracted_data[doc_idx].extend(fallback_results)

        return per_document_extracted_data

    async def extract(
        self,
        question: str,
        schema: Tables,
        documents: list[Document],
        metadata: dict,
        task_guidelines: str,
    ) -> dict:
        extraction_start_time = time.time()
        extract_payload = deepcopy(self.model_config["extract_schema"])
        num_samples = self.config.get("num_samples_per_chunk", 1)
        if num_samples > 1:
            extract_payload["temperature"] = extract_payload.get("temperature", 0.7)

        extract_schema_chain = self.create_extract_schema_chain(**extract_payload)

        extract_config = self.model_config["extract_schema"]
        logger.info(
            f"Using extraction model={extract_config.get('model', 'N/A')} "
            f"max_tokens={extract_config.get('max_tokens', 'N/A')} "
            f"temperature={extract_config.get('temperature', 'N/A')}"
        )
        if self.config.get("is_relevant_chunk", False):
            is_relevant_config = self.model_config.get("is_relevant_chunk", {})
            logger.info(
                f"Using is_relevant_chunk model={is_relevant_config.get('model', 'N/A')} "
                f"max_tokens={is_relevant_config.get('max_tokens', 'N/A')} "
                f"temperature={is_relevant_config.get('temperature', 'N/A')}"
            )

        schema_repr = prepare_schema_repr(schema)

        if self.config.get("enable_extraction_guidelines", False):
            extraction_guidelines_repr = self._format_extraction_guidelines(schema)
            if extraction_guidelines_repr:
                logger.info("=" * 60)
                logger.info("EXTRACTION GUIDELINES FROM SCHEMA:")
                logger.info("=" * 60)
                for line in extraction_guidelines_repr.split("\n"):
                    logger.info(line)
                logger.info("=" * 60)
            else:
                logger.info("No extraction guidelines specified in schema")
        else:
            extraction_guidelines_repr = None
            logger.info("Extraction guidelines disabled by config")

        per_document_extracted_data: list[list[dict]] = []
        per_document_irrelevant_chunks: list[set[int]] = []
        total_chunks = 0
        successful_extractions = 0
        failed_extractions = 0
        retry_attempts = 0

        extraction_batch_size = self.config.get("extraction_batch_size", 1)
        logger.info(f"Using extraction batch size: {extraction_batch_size} documents")

        for batch_start in range(0, len(documents), extraction_batch_size):
            batch_end = min(batch_start + extraction_batch_size, len(documents))
            batch_documents = documents[batch_start:batch_end]
            batch_doc_ids = list(range(batch_start, batch_end))
            logger.info(f"Processing document batch {batch_start}-{batch_end} ({len(batch_documents)} documents)")

            all_batch_tasks = []
            task_to_doc_chunk_sample: list[tuple[int, int, int]] = []

            for batch_doc_idx, document in enumerate(batch_documents):
                for chunk_id, chunk in enumerate(document.chunks):
                    total_chunks += 1
                    for sample_idx in range(num_samples):
                        all_batch_tasks.append(
                            self._extract_chunk_single(
                                question_id=metadata.get("question_id", None),
                                extract_chain=extract_schema_chain,
                                schema_repr=schema_repr,
                                extraction_guidelines_repr=extraction_guidelines_repr,
                                document=document,
                                chunk=chunk,
                                chunk_id=chunk_id,
                                question=question,
                                task_guidelines=task_guidelines,
                            )
                        )
                        task_to_doc_chunk_sample.append((batch_doc_idx, chunk_id, sample_idx))

            try:
                raw_results = await asyncio.gather(*all_batch_tasks, return_exceptions=True)

                per_doc_chunk_samples: list[dict[int, list[Any]]] = [defaultdict(list) for _ in batch_documents]
                per_doc_irrelevant: list[set[int]] = [set() for _ in batch_documents]

                for task_idx, result in enumerate(raw_results):
                    batch_doc_idx, chunk_id, _sample_idx = task_to_doc_chunk_sample[task_idx]
                    if result == "irrelevant":
                        per_doc_irrelevant[batch_doc_idx].add(chunk_id)
                        continue
                    per_doc_chunk_samples[batch_doc_idx][chunk_id].append(result)

                batch_extracted_data: list[list[Any]] = [[] for _ in batch_documents]
                for batch_doc_idx, document in enumerate(batch_documents):
                    doc_samples = per_doc_chunk_samples[batch_doc_idx]
                    irrelevant_chunks = per_doc_irrelevant[batch_doc_idx]
                    for chunk_id in range(len(document.chunks)):
                        samples = doc_samples.get(chunk_id, [])
                        if not samples and chunk_id in irrelevant_chunks:
                            batch_extracted_data[batch_doc_idx].append(None)
                            continue
                        if not samples:
                            batch_extracted_data[batch_doc_idx].append(None)
                            continue
                        if len(samples) == 1:
                            batch_extracted_data[batch_doc_idx].append(samples[0])
                        else:
                            merged = self._merge_chunk_results(
                                samples,
                                base_metadata={
                                    "chunk_id": chunk_id,
                                    "document_name": document.document_name,
                                },
                            )
                            batch_extracted_data[batch_doc_idx].append(
                                {rel: [row.model_dump(by_alias=True) for row in rows] for rel, rows in merged.items()}
                            )
                    per_doc_irrelevant[batch_doc_idx] = irrelevant_chunks

                for batch_doc_idx, document in enumerate(batch_documents):
                    doc_id = batch_doc_ids[batch_doc_idx]
                    extracted_data = batch_extracted_data[batch_doc_idx]
                    irrelevant_chunk_indexes = per_doc_irrelevant[batch_doc_idx]

                    successful_extractions, failed_extractions, retry_attempts = await self.handle_failed_extractions(
                        question_id=metadata.get("question_id", None),
                        question=question,
                        extracted_data=extracted_data,
                        metadata=metadata,
                        schema=schema,
                        document=document,
                        schema_repr=schema_repr,
                        successful_extractions=successful_extractions,
                        failed_extractions=failed_extractions,
                        retry_attempts=retry_attempts,
                        irrelevant_chunk_indexes=irrelevant_chunk_indexes,
                    )

                    non_none_count = sum(1 for x in extracted_data if x is not None)
                    logger.info(
                        f"Converting {len(extracted_data)} chunks to JSON ({non_none_count} non-None) "
                        f"for document {document.document_name}"
                    )

                    json_repr = self.convert_extracted_data_to_json(extracted_data, document, doc_id)

                    await self.checkback_extracted_data(
                        question_id=metadata.get("question_id", None),
                        extracted_data=json_repr,
                        document=document,
                        doc_id=doc_id,
                        schema=schema,
                        question=question,
                    )
                    per_document_extracted_data.append(json_repr)
                    per_document_irrelevant_chunks.append(irrelevant_chunk_indexes)

            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error extracting schema for document batch {batch_start}-{batch_end}: {e}")
                logger.error(traceback.format_exc())
                for document in batch_documents:
                    metadata["errors"].append(
                        {"stage": "extraction_batch", "error": str(e), "document": document.document_name}
                    )
                failed_extractions += len(all_batch_tasks)
                for document in batch_documents:
                    per_document_extracted_data.append([None] * len(document.chunks))
                    per_document_irrelevant_chunks.append(set())

        metadata["extraction"]["chunks_processed"] = total_chunks
        metadata["extraction"]["successful_extractions"] = successful_extractions
        metadata["extraction"]["failed_extractions"] = failed_extractions
        metadata["extraction"]["retry_attempts"] = retry_attempts
        metadata["extraction"]["extraction_time"] = time.time() - extraction_start_time
        metadata["extraction"]["success_rate"] = successful_extractions / total_chunks if total_chunks > 0 else 0

        # Run fallback extraction for documents with missing data
        per_document_extracted_data = await self._run_extraction_fallback(
            question=question,
            schema=schema,
            documents=documents,
            metadata=metadata,
            task_guidelines=task_guidelines,
            per_document_extracted_data=per_document_extracted_data,
            per_document_irrelevant_chunks=per_document_irrelevant_chunks,
        )

        absent_decisions = await self._decide_absent_table_handling(
            per_document_extracted_data,
            documents,
            schema,
            question,
        )
        final_extracted_data = self.finalize_tables(
            per_document_extracted_data,
            documents,
            schema,
            absent_decisions,
        )

        try:
            experiment_id = SlidersGlobal.experiment_id or "unknown_experiment"
            question_id = metadata.get("question_id") or "unknown_question"
            safe_question_id = str(question_id).replace(os.sep, "_")
            base_dir = os.environ.get("SLIDERS_RESULTS", ".")
            os.makedirs(base_dir, exist_ok=True)
            for table_name, rows in final_extracted_data.items():
                file_name = f"{experiment_id}_{safe_question_id}_{table_name}.json"
                file_path = os.path.join(base_dir, file_name)
                with open(file_path, "w") as f:
                    json.dump(rows, f, indent=2, default=str)
        except Exception as e:  # pragma: no cover - best effort logging
            logger.error(f"Failed to save finalized extraction tables: {e}")
            logger.error(traceback.format_exc())

        return final_extracted_data
