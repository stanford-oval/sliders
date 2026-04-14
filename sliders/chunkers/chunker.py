import re
from copy import deepcopy
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter, RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from sliders.document import Document


class CustomMarkdownTextSplitter:
    def __init__(
        self,
        strip_headers: bool = False,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        format_chunks: bool = True,
    ):
        self.strip_headers = strip_headers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.format_chunks = format_chunks

        self.md_splitter = ExperimentalMarkdownSyntaxTextSplitter(
            strip_headers=self.strip_headers,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".\n", "\n"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.headers = {
            "Header 1": "#",
            "Header 2": "##",
            "Header 3": "###",
            "Header 4": "####",
            "Header 5": "#####",
            "Header 6": "######",
        }

    def _format_chunks(self, chunks: List[LangchainDocument]) -> List[LangchainDocument]:
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            first_headers = []
            for key, values in chunk.metadata.items():
                if key not in self.headers:
                    continue
                if isinstance(values, list):
                    first_headers.extend([self.headers[key] + " " + value for value in values])
                else:
                    first_headers.append(self.headers[key] + " " + values)

            sorted_headers = sorted(first_headers, key=lambda x: len(x.split(" ")[0]))
            sorted_headers = sorted(first_headers, key=lambda x: len(x.split(" ")[0]))

            headers_in_chunk = re.findall(r"(#+.*)", chunk.page_content)
            text = ""
            if len(headers_in_chunk):
                for first_header in sorted_headers:
                    if headers_in_chunk[0] != first_header:
                        text += first_header + "\n...\n"
                    else:
                        break
            else:
                text += "\n".join([header for header in sorted_headers]) + "\n...\n"
            text += chunk.page_content
            for key, values in chunk.metadata.items():
                if isinstance(values, list):
                    chunk.metadata[key] = " > ".join(values)

            formatted_chunks.append(LangchainDocument(page_content=text, metadata=chunk.metadata))
        return formatted_chunks

    def _merge_chunks(self, chunks: List[LangchainDocument]) -> List[LangchainDocument]:
        merged_chunks = []
        for chunk in chunks:
            if len(merged_chunks) == 0:
                merged_chunks.append(chunk)
            elif len(merged_chunks[-1].page_content) + len(chunk.page_content) <= self.chunk_size:
                merged_chunks[-1].page_content += "\n\n" + chunk.page_content
                if merged_chunks[-1].metadata is None:
                    merged_chunks[-1].metadata = chunk.metadata
                else:
                    for key, value in chunk.metadata.items():
                        if key not in merged_chunks[-1].metadata:
                            if isinstance(value, list):
                                merged_chunks[-1].metadata[key] = value
                            else:
                                merged_chunks[-1].metadata[key] = [value]
                        else:
                            if isinstance(merged_chunks[-1].metadata[key], list):
                                if isinstance(value, list):
                                    merged_chunks[-1].metadata[key].extend(value)
                                else:
                                    merged_chunks[-1].metadata[key].append(value)
                            else:
                                if isinstance(value, list):
                                    merged_chunks[-1].metadata[key] = [
                                        merged_chunks[-1].metadata[key],
                                        *value,
                                    ]
                                else:
                                    merged_chunks[-1].metadata[key] = [
                                        merged_chunks[-1].metadata[key],
                                        value,
                                    ]
            else:
                merged_chunks.append(chunk)
        return merged_chunks

    def split_text(self, text: str) -> List[LangchainDocument]:
        text = text.replace("<!-- image -->", "<!-- image -->\n")
        md_splits = self.md_splitter.split_text(text)
        text_splits = self.text_splitter.split_documents(md_splits)

        merged_chunks = self._merge_chunks(text_splits)

        if self.format_chunks:
            return self._format_chunks(merged_chunks)
        return merged_chunks


class Chunker:
    def __init__(
        self,
        chunk_size: int = 4000,
        overlap_size: int = 200,
        splitter_class=None,
        splitter_kwargs=None,
        format_chunks: bool = True,
    ):
        """Initialize the basic Chunker.

        Args:
            chunk_size: Number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
            splitter_class: Class to use for splitting text
            splitter_kwargs: Keyword arguments to pass to the splitter
            format_chunks: Whether to format chunks with headers
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        if splitter_class is None:
            self.splitter = CustomMarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                format_chunks=format_chunks,
            )
        else:
            splitter_kwargs = splitter_kwargs or {}
            self.splitter = splitter_class(chunk_size=chunk_size, chunk_overlap=overlap_size, **splitter_kwargs)

    def chunk_document(self, document: "Document") -> List["Document"]:
        """Split document into basic chunks without context.

        Args:
            document: Document to split

        Returns:
            List of Document objects containing chunks
        """
        # Get base chunks
        chunks = self.splitter.split_text(document.content)

        # Create new documents without context
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            doc = deepcopy(document)
            doc.content = chunk.page_content if hasattr(chunk, "page_content") else chunk
            doc.metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            chunked_docs.append(doc)

        return chunked_docs

    def chunk_text(self, text: str, replace_tables: bool = True, tag_to_table: dict = None) -> List[dict]:
        chunks = self.splitter.split_text(text)
        if replace_tables:
            for chunk in chunks:
                for tag, table in tag_to_table.items():
                    chunk.page_content = chunk.page_content.replace(tag, table)

        new_chunks = [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
        return new_chunks
