from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

from sliders.callbacks.logging import LoggingHandler
from sliders.document import Document
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.llm_models import RephrasedQuestion
from sliders.log_utils import logger

CURRENT_DIR = Path(__file__).parent


@dataclass
class ComponentQuestionSet:
    """Container for component-aware rephrased questions."""

    original_question: str
    schema_question: str
    extraction_question: str
    merge_question: str
    answer_question: str
    reasoning: dict[str, str] = field(default_factory=dict)
    fallback_components: set[str] = field(default_factory=set)

    def as_list(self) -> list[str]:
        """Return questions in pipeline order."""
        return [
            self.schema_question,
            self.extraction_question,
            self.merge_question,
            self.answer_question,
        ]

    def as_dict(self) -> dict[str, dict[str, str | None]]:
        """Return mapping of component to question and reasoning."""
        return {
            "schema": {
                "question": self.schema_question,
                "reasoning": self.reasoning.get("schema"),
            },
            "extraction": {
                "question": self.extraction_question,
                "reasoning": self.reasoning.get("extraction"),
            },
            "merge": {
                "question": self.merge_question,
                "reasoning": self.reasoning.get("merge"),
            },
            "answer": {
                "question": self.answer_question,
                "reasoning": self.reasoning.get("answer"),
            },
        }


class QuestionRephraser:
    """Generate component-aware rewrites of an input question."""

    DEFAULT_PROMPT_FILE = "sliders/rephrase_question_component.prompt"
    COMPONENT_ORDER: tuple[str, ...] = ("schema", "extraction", "merge", "answer")

    def __init__(self, config: dict[str, Any] | None, model_config: dict[str, Any]):
        self.config = config or {}
        self.model_config = model_config
        self.prompt_file = self.config.get("prompt_file", self.DEFAULT_PROMPT_FILE)

        # Load extraction rephrasing guidelines library (same as schema guidelines)
        self.guidelines_library = None
        if self.config.get("library_of_guidelines_path", None):
            self.guidelines_library = self._load_guidelines_library()

        # Store the LLM client for later use
        llm_kwargs = self._get_llm_kwargs()
        self.llm_client = get_llm_client(**llm_kwargs)

        # Initialize chains dict (will be built per-call with document type)
        self._chains: dict[str, Any] = {}

    def _load_guidelines_library(self) -> dict:
        """Load the guidelines library (same as schema guidelines)."""
        library_path = Path(CURRENT_DIR) / "../" / self.config["library_of_guidelines_path"]
        with open(library_path, "r") as f:
            return json.load(f)

    def _get_llm_kwargs(self) -> dict[str, Any]:
        llm_kwargs = self.model_config.get("rephrase_question")
        if llm_kwargs is None:
            raise ValueError("Missing 'rephrase_question' model configuration for QuestionRephraser.")
        return llm_kwargs

    def _build_component_templates(
        self, document_type: str | None = None, question_type: str | None = None
    ) -> dict[str, Any]:
        base_template = load_fewshot_prompt_template(
            template_file=self.prompt_file,
            template_blocks=[],
        )
        templates = {}
        for component, spec in self._component_spec(document_type, question_type).items():
            templates[component] = base_template.partial(
                objective_name=spec["objective_name"],
                objective_description=spec["objective_description"],
                guidelines_text=self._format_guidelines(spec["guidelines"]),
                component_hint=spec["component_hint"],
            )
        return templates

    @staticmethod
    def _format_guidelines(guidelines: Iterable[str]) -> str:
        items = [f"- {guideline}" for guideline in guidelines]
        return "\n".join(items)

    def _component_spec(
        self, document_type: str | None = None, question_type: str | None = None
    ) -> dict[str, dict[str, str | list[str]]]:
        # Base extraction guidelines (without the relationship extraction one)
        extraction_base_guidelines = [
            "Request exhaustive data that matches the schema fields.",
            "Avoid positional directives like 'first', 'second', or 'overall totals'—extract everything that fits.",
            "Assume each chunk lacks global awareness; instruct the extractor to capture all relevant evidence it sees.",
            "Provide all the document names that are provided for the task.",
            "If labelling is required, then provide the list of labels that are available.",
            "For classification questions where the original question provides label definitions (e.g., 'label1: definition, label2: definition'), include these definitions in the rephrased extraction question to guide the extractor.",
        ]

        # Add document type-specific extraction guidelines if library is loaded
        if self.guidelines_library and document_type:
            doc_extraction_guidelines = (
                self.guidelines_library["document_type"]
                .get(document_type, {})
                .get("extraction_rephrasing_guidelines", None)
            )
            if doc_extraction_guidelines:
                extraction_base_guidelines.extend(doc_extraction_guidelines)
                logger.info(
                    f"Injected {len(doc_extraction_guidelines)} extraction guideline(s) for document_type='{document_type}'"
                )
            else:
                logger.info(f"No extraction guidelines to inject for document_type='{document_type}'")

        # Add question type-specific extraction guidelines if library is loaded
        if self.guidelines_library and question_type:
            ques_extraction_guidelines = (
                self.guidelines_library["question_type"]
                .get(question_type, {})
                .get("extraction_rephrasing_guidelines", None)
            )
            if ques_extraction_guidelines:
                extraction_base_guidelines.extend(ques_extraction_guidelines)
                logger.info(
                    f"Injected {len(ques_extraction_guidelines)} extraction guideline(s) for question_type='{question_type}'"
                )
            else:
                logger.info(f"No extraction guidelines to inject for question_type='{question_type}'")

        return {
            "schema": {
                "objective_name": "Schema Generation",
                "objective_description": (
                    "Draft an extraction schema that contains every attribute needed to answer the original question. "
                    "Keep the schema document-agnostic and anticipate the fields required across all documents."
                ),
                "guidelines": [
                    "Generalize: avoid document-specific values or enumerations.",
                    "Make the schema expressive enough to capture every detail necessary for the final answer.",
                    "Anticipate related attributes (e.g., include units, dates, parties, totals when relevant).",
                ],
                "component_hint": (
                    "Write the question like guidance for a schema designer. "
                    "Focus on what information should be captured, not how to process the documents."
                ),
            },
            "extraction": {
                "objective_name": "Chunk Extraction",
                "objective_description": (
                    "Guide chunk-level extraction so each chunk yields all rows needed for the schema, even if context "
                    "spans multiple chunks."
                ),
                "guidelines": extraction_base_guidelines,
                "component_hint": (
                    "Phrase the question so the extractor focuses on gathering every possible row and field that matches "
                    "the schema requirements."
                ),
            },
            "merge": {
                "objective_name": "Table Merging",
                "objective_description": (
                    "Combine chunk-level rows into clean tables aligned with the schema. If needed, resolve conflicts, "
                    "deduplicating entries, and aggregating when appropriate using the quote and reasoning provided."
                ),
                "guidelines": [
                    "Emphasize preserving data fidelity with the schema fields.",
                    "Provide all the document names that are provided for the task."
                    "If labelling is required, then provide the list of labels that are available.",
                    "Do not instruct how to write the final answer, for example do not ask to write as json. The output should be a final table, which can be used to frame the final answer.",
                    "The merge should only consolidate and clean the extracted data (deduplicate, resolve conflicts, combine partial information). Do NOT ask it to compute complex derived results or the final answer itself.",
                    "For tasks requiring multi-step computation (e.g., finding longest chains, computing rankings, complex aggregations), ask the merge to preserve the base relationships in normalized form—let the answer stage perform the computation.",
                    "Do NOT ask the merge to filter or focus on specific entities from the question (e.g., 'for paper X, show...'). The merge should consolidate ALL extracted data—let the answer stage do the filtering.",
                    "Do NOT ask the merge to reshape data into aggregated formats (e.g., 'one row per entity with lists of related items'). Keep relationships in normalized form (one row per relationship)—let the answer stage aggregate if needed.",
                ],
                "component_hint": (
                    "Frame the question so the merger understands what the consolidated table should contain once all "
                    "chunks are combined. Focus on data quality and consolidation, not computing the final answer."
                ),
            },
            "answer": {
                "objective_name": "Answer Generation",
                "objective_description": (
                    "Produce the final natural-language answer that satisfies the user's request using the merged tables."
                ),
                "guidelines": [
                    "Keep the question faithful to the original intent.",
                    "Remove formatting, JSON, or pipeline-specific instructions.",
                    "Clarify ambiguous phrasing if needed while keeping scope unchanged.",
                ],
                "component_hint": (
                    "This is usually close to the original question—clean it up so it is ready for the answer generator."
                ),
            },
        }

    @staticmethod
    def _format_document_names(documents: list[Document]) -> str:
        if not documents:
            return "No documents provided."
        return "\n".join(f"- {doc.document_name or 'Unknown Document'}" for doc in documents)

    @staticmethod
    def _format_document_descriptions(documents: list[Document]) -> str:
        if not documents:
            return "No descriptions provided."
        lines = []
        for doc in documents:
            description = doc.description or "No description available."
            lines.append(f"- {doc.document_name or 'Unknown Document'}: {description}")
        return "\n".join(lines) if lines else "No descriptions provided."

    async def rephrase(
        self,
        question: str,
        documents: list[Document],
        metadata: dict,
        document_type: str | None = None,
        question_type: str | None = None,
    ) -> ComponentQuestionSet:
        """Generate component-aware questions.

        Args:
            question: The original question to rephrase
            documents: List of documents
            metadata: Metadata dictionary
            document_type: Document type classification from schema generation (e.g., 'story', 'dataset', 'policy', 'others')
            question_type: Question type classification from schema generation (e.g., 'simple', 'ordering_questions', 'multiple_choice')
        """
        document_names = self._format_document_names(documents)
        document_descriptions = self._format_document_descriptions(documents)

        shared_payload = {
            "question": question,
            "document_list": document_names,
            "document_descriptions": document_descriptions,
        }

        schema_question = question
        extraction_question = question
        merge_question = question
        answer_question = question
        reasoning: Dict[str, str] = {}
        fallback_components: set[str] = set()

        # Build component templates with document/question type
        component_templates = self._build_component_templates(document_type, question_type)
        chains = {}
        for component, template in component_templates.items():
            chains[component] = template | self.llm_client.with_structured_output(RephrasedQuestion)

        for component in self.COMPONENT_ORDER:
            chain = chains[component]
            handler = LoggingHandler(
                prompt_file=f"{self.prompt_file}::{component}",
                metadata={
                    "question": question,
                    "documents": document_names,
                    "component": component,
                    "stage": "rephrase_question",
                    "question_id": metadata.get("question_id"),
                },
            )

            try:
                result: RephrasedQuestion = await chain.ainvoke(
                    shared_payload,
                    config={"callbacks": [handler]},
                )
            except Exception as exc:
                logger.exception(f"Failed to rephrase question for component '{component}': {exc}")
                metadata["errors"].append(
                    {
                        "stage": "rephrase_question",
                        "component": component,
                        "error": str(exc),
                    }
                )
                fallback_components.add(component)
                reasoning[component] = f"Fallback to original question due to error: {exc}"
                continue

            question_text = (result.question or "").strip()
            reasoning[component] = (result.reasoning or "").strip()

            if not question_text:
                fallback_components.add(component)
                reasoning[component] = (
                    reasoning[component] or "Fallback to original question because no rewrite was returned."
                )
                continue

            if component == "schema":
                schema_question = question_text
            elif component == "extraction":
                extraction_question = question_text
            elif component == "merge":
                merge_question = question_text
            elif component == "answer":
                answer_question = question_text

        return ComponentQuestionSet(
            original_question=question,
            schema_question=schema_question,
            extraction_question=extraction_question,
            merge_question=merge_question,
            answer_question=answer_question,
            reasoning=reasoning,
            fallback_components=fallback_components,
        )
