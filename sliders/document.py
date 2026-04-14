import re
import os
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


from sliders.callbacks.logging import LoggingHandler
from sliders.chunkers.chunker import Chunker
from sliders.markdown_utils import find_table_in_markdown_doc, parse_markdown, renderer
from sliders.llm_models import DocumentDescriptions, DocumentTitle
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template


@dataclass
class Document:
    content: str
    processed_content: str
    tables: dict[str, str]
    chunks: list[dict[str, str]]
    document_name: str
    description: str
    file_path: str

    @classmethod
    async def from_file_path(
        cls,
        text_path: str,
        description: str,
        document_name=None,
        chunker: Chunker | None = None,
        tables_json_path: Optional[str] = None,
        replace_tables_tag: bool = True,
        **kwargs,
    ):
        with open(text_path, "r") as f:
            text = f.read()
        return await cls.from_plain_text(
            text, description, document_name, chunker, tables_json_path, text_path, replace_tables_tag, **kwargs
        )

    @classmethod
    async def from_plain_text(
        cls,
        plain_text: str,
        description: str,
        document_name=None,
        chunker: Chunker | None = None,
        tables_json_path: Optional[str] = None,
        file_path: Optional[str] = None,
        replace_tables_tag: bool = True,
        **kwargs,
    ):
        if chunker is None:
            chunker = Chunker(chunk_size=kwargs.get("chunk_size", 4000), overlap_size=kwargs.get("overlap_size", 0))

        processed_text = plain_text
        tag_to_table: Dict[str, str] = {}

        # If a tables JSON path is provided, try to load and apply it
        if tables_json_path is not None:
            processed_text, tag_to_table = _build_tagged_text_and_mapping_from_tables_json(plain_text, tables_json_path)

        # Replace tags in chunks only if we actually created any tag mappings
        chunks = chunker.chunk_text(
            processed_text,
            replace_tables=replace_tables_tag and bool(tag_to_table),
            tag_to_table=tag_to_table,
        )
        if document_name is None:
            document_name = await get_doc_title_from_text(chunks[0]["content"], file_path)
        print(f"Extracting {len(chunks)} chunks from {document_name}")

        return cls(
            content=plain_text,
            description=description,
            processed_content=processed_text,
            tables=tag_to_table,
            chunks=chunks,
            document_name=document_name,
            file_path=file_path,
        )

    @classmethod
    async def from_markdown(
        cls,
        markdown_path: str,
        description: str,
        replace_with_summary=False,
        document_name=None,
        chunker: Chunker | None = None,
        replace_tables_tag: bool = True,
        title_model: str = "gpt-4.1",
        table_summary_model: str = "gpt-4.1-mini",
        **kwargs,
    ):
        with open(markdown_path, "r") as f:
            markdown = f.read()

        if not document_name:
            document_name = await get_doc_title_from_markdown(markdown_path, model=title_model)

        parsed = parse_markdown(markdown)

        tables = []
        find_table_in_markdown_doc(parsed, tables)

        texts = []
        tag_to_table = {}
        for i, table in enumerate(tables):
            tag = f"<table id='{i}'>"
            texts.append(tag)
            tag_to_table[tag] = renderer.render(table)

        if replace_with_summary:
            texts = await summarize_tables(tables, model=table_summary_model)

        markdown_text = replace_tables(markdown, texts)

        if chunker is None:
            chunker = Chunker(chunk_size=kwargs.get("chunk_size", 4000), overlap_size=kwargs.get("overlap_size", 0))

        chunks = chunker.chunk_text(markdown_text, replace_tables=replace_tables_tag, tag_to_table=tag_to_table)

        return cls(
            content=markdown,
            description=description,
            processed_content=markdown_text,
            tables=tag_to_table,
            chunks=chunks,
            document_name=document_name,
            file_path=markdown_path,
        )


def replace_tables(markdown_text, table_texts: list[str]):
    table_pattern = re.compile(r"((\|.*\|.*\n)+)")
    matches = list(table_pattern.finditer(markdown_text))
    new_markdown = markdown_text
    for match, table_text in zip(reversed(matches), reversed(table_texts)):
        start, end = match.span()
        new_markdown = new_markdown[:start] + table_text + "\n" + new_markdown[end:]
    return new_markdown


async def summarize_tables(tables, model: str = "gpt-4.1-mini"):
    llm_client = get_llm_client(model=model, temperature=0.0)
    summarize_tables_template = load_fewshot_prompt_template(
        template_file="summarize_tables.prompt",
        template_blocks=[
            (
                "instruction",
                """Summarize the table given by the user in less than 50 words. 

    The summary should focus on what information is present in the table, and NOT on data trends in the table.
    """,
            ),
            ("input", "{{table}}"),
        ],
    )
    summarize_tables_chain = summarize_tables_template | llm_client

    inputs = [{"table": renderer.render(table)} for table in tables]

    summaries = await summarize_tables_chain.abatch(inputs)

    summaries = [f"<table id='{i}'>{summary.content}</table>" for i, summary in enumerate(summaries)]

    return summaries


def table_to_list(table_node):
    headers = [cell.children[0].children for cell in table_node.head.children if len(cell.children) > 0]
    rows = []
    for row in table_node.children:
        if hasattr(row, "children"):
            rows.append([cell.children[0].children for cell in row.children if len(cell.children) > 0])
    return headers, rows


async def get_doc_title_from_markdown(markdown_file_path: str, model: str = "gpt-4.1"):
    doc_name = None
    first_lines = ""
    with open(markdown_file_path) as f:
        for _ in range(5):
            line = f.readline()
            if line.startswith("#"):
                doc_name = line.lstrip("#").strip()
                break

                first_lines += line
    if not doc_name:
        doc_name = await get_doc_title_from_text(first_lines, markdown_file_path, model=model)
        if not doc_name:
            doc_name = markdown_file_path.split("/")[-1]
    return doc_name


async def get_doc_title_from_text(first_chunk_content: str, file_path: str, model: str = "gpt-4.1"):
    if len(first_chunk_content.strip()) < 10:
        return None

    logging_handler = LoggingHandler(
        prompt_file="extract_title.prompt",
        metadata={
            "file_name": file_path,
            "text": first_chunk_content[:500] + "...",
        },
    )

    try:
        llm_client = get_llm_client(model=model, temperature=0.0, max_tokens=100)
        extract_title_template = load_fewshot_prompt_template(
            template_file="extract_title.prompt",
            template_blocks=[
                (
                    "instruction",
                    """Extract or Generate a concise title for this given text (this is an excerpt from the beginning of the file).

    Look at the start of the text, and decide if the title is already present in the text. If it is, extract it. If it is not, generate a new title.

    If you have to generate a new title, the title should be:
    1. Less than 10 words
    2. Descriptive of the main topic
    3. In plain text (no markdown)
                    
    Use the same language as the text to extract a reasonable title.""",
                ),
                (
                    "input",
                    """
                File: {{file_name}}
                Text content preview: {{text}}
                """,
                ),
            ],
        )
        extract_title_chain = extract_title_template | llm_client.with_structured_output(DocumentTitle)

        # Extract title using LLM
        default_title = file_path.split("/")[-1]
        title = await extract_title_chain.ainvoke(
            {"file_name": default_title, "text": first_chunk_content[:500] + "..."},
            config={"callbacks": [logging_handler]},
        )

        logger.info(f"Document Title: {default_title} -> {title.title}")
        return title.title
    except Exception:
        try:
            llm_client = get_llm_client(model=model, temperature=0.7, max_tokens=100)
            extract_title_template = load_fewshot_prompt_template(
                template_file="extract_title.prompt",
                template_blocks=[
                    (
                        "instruction",
                        """Extract or Generate a concise title for this given text (this is an excerpt from the beginning of the file).

        Look at the start of the text, and decide if the title is already present in the text. If it is, extract it. If it is not, generate a new title.

        If you have to generate a new title, the title should be:
        1. Less than 10 words
        2. Descriptive of the main topic
        3. In plain text (no markdown)
                        
        Use context clues from the text context formatting, the file name, and the text excerpt itself to extract a reasonable title.""",
                    ),
                    (
                        "input",
                        """
                    File: {{file_name}}
                    Text content preview: {{text}}
                    """,
                    ),
                ],
            )
            extract_title_chain = extract_title_template | llm_client.with_structured_output(DocumentTitle)

            # Extract title using LLM
            default_title = file_path.split("/")[-1]
            title = await extract_title_chain.ainvoke(
                {"file_name": default_title, "text": first_chunk_content[:500] + "..."},
                config={"callbacks": [logging_handler]},
            )

            logger.info(f"Document Title: {default_title} -> {title.title}")
            return title.title
        except Exception as e:
            logger.error(f"Error extracting title: {e}. Using default title: {default_title}")
            return default_title


async def contextualize_document_metadata(
    documents: list[Document],
    question: str,
    model: str = "gpt-4.1-mini",
) -> list[Document]:
    if not documents:
        return documents

    logging_handler = LoggingHandler(
        prompt_file="regenerate_descriptions.prompt",
        metadata={
            "documents": [doc.document_name for doc in documents],
            "question": question,
            "stage": "contextualize_document_metadata",
        },
    )

    llm_client = get_llm_client(model=model, temperature=0.0)
    regenerate_descriptions_template = load_fewshot_prompt_template(
        template_file="regenerate_descriptions.prompt",
        template_blocks=[
            (
                "instruction",
                """Given the document titles, the original general document descriptions, and the first part of the document, generate new descriptions for each document that:
1. Clearly state what kind of document it is; use the original description as a starting point.
2. Differentiate it from the other documents listed.
3. Stay concise (1 phrase), avoiding analysis, evaluation, or commentary on content.
4. The description should be in the same language as the document.
5. Try to use keywords in the document content and use them to generate the description.

...                
Format the response as a JSON object with a 'descriptions' field containing an array of strings, one description per document.
Example: {"descriptions": ["Description 1", "Description 2", ...]}
                """,
            ),
            (
                "input",
                """
**Document information:**
{{document_information}}
""",
            ),
        ],
    )
    regenerate_descriptions_chain = regenerate_descriptions_template | llm_client.with_structured_output(
        DocumentDescriptions
    )

    try:
        document_information = "\n".join(
            [
                f"# {doc.document_name} # Document Content\n{doc.chunks[0]['content'][:500] + '...' if doc.chunks else ''}\n ----------------------------------------"
                for doc in documents
            ]
        )

        result = await regenerate_descriptions_chain.ainvoke(
            {
                "document_information": document_information,
            },
            config={"callbacks": [logging_handler]},
        )

        if not result or not result.descriptions or len(result.descriptions) != len(documents):
            logger.warning("Invalid or incomplete descriptions generated. Keeping original descriptions.")
            return documents

        # Update descriptions if we got valid results
        for doc, new_desc in zip(documents, result.descriptions):
            if new_desc:
                logger.info(
                    f"Document: {doc.document_name}\nOriginal description: {doc.description}\nNew description: {new_desc}"
                )
                doc.description = new_desc

        logger.info(
            f"New document descriptions: {result.descriptions[:3]} {'...' if len(result.descriptions) > 3 else ''}"
        )

    except Exception as e:
        logger.error(f"Error generating new descriptions: {e}. Keeping original descriptions.")

    return documents


def _build_tagged_text_and_mapping_from_tables_json(
    plain_text: str, tables_json_path: str
) -> Tuple[str, Dict[str, str]]:
    if not tables_json_path or not os.path.exists(tables_json_path):
        return plain_text, {}

    with open(tables_json_path, "r") as f:
        data = json.load(f)

    tables = data.get("tables", []) if isinstance(data, dict) else []
    if not tables:
        return plain_text, {}

    lines = plain_text.split("\n")
    tag_to_table = {}
    replacements = []

    for i, t in enumerate(tables):
        first_line_index = t.get("first_line_index")
        last_line_index = t.get("last_line_index")

        if first_line_index is None or last_line_index is None:
            continue

        # Slice table text *without trimming away context lines*
        table_text = "\n".join(lines[first_line_index : last_line_index + 1])
        table_notes = t.get("table_notes") or ""
        tag = f"<table id='{i}'>"

        # Store *exact content* for later reconstruction
        content = f"Table {i + 1} NOTE: {table_notes}\n\n{table_text}\n\n" if table_notes else table_text
        tag_to_table[tag] = content

        replacements.append((first_line_index, last_line_index, tag))

    # Apply replacements from the end (to keep indices valid)
    for first, last, tag in sorted(replacements, key=lambda x: x[0], reverse=True):
        lines[first : last + 1] = [tag]

    return "\n".join(lines), tag_to_table
