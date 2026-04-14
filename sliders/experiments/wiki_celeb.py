import os
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from sliders.document import contextualize_document_metadata
from sliders.baselines import System
from sliders.document import Document
from sliders.globals import SlidersGlobal
from sliders.log_utils import logger
from sliders.experiments.base import Experiment


def log_wiki_celeb_results(result):
    if "error" in result:
        logger.info(f"Error: {result['error']}")
    else:
        logger.info(f"Question: {result['question']}")
        logger.info(f"Predicted Answer: {result['predicted_answer']}")


class WikiCeleb(Experiment):
    """Multi-document QA driver used by both the WikiCeleb100 benchmark and :mod:`sliders.run`.

    Reads one question per line from ``questions_path`` and treats every
    ``.md`` file under ``files_dir`` as part of a single corpus. No gold
    answers are required, so the driver does not run evaluation.
    """

    def __init__(self, config: dict):
        self.config = config

        questions_path = self.config.get("questions_path")
        files_dir = self.config.get("files_dir")

        if questions_path is None or files_dir is None:
            raise ValueError(
                "WikiCeleb requires both 'questions_path' and 'files_dir' in its config. "
                "When running via sliders.run.run_sliders, these are set automatically."
            )

        self.questions = self._load_questions(questions_path)
        self.files_dir = files_dir
        self.documents_cache: list[Document] | None = None

    def _load_questions(self, questions_path: str) -> list[dict]:
        """Parse ``questions_path`` into ``[{"id", "question"}, ...]``."""
        questions = []
        with open(questions_path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    questions.append(
                        {
                            "id": f"wiki_celeb_{idx + 1:03d}",
                            "question": line,
                        }
                    )
        logger.info(f"Loaded {len(questions)} questions from {questions_path}")
        return questions

    def _get_all_markdown_files(self) -> list[str]:
        """Return all ``.md`` file paths under ``self.files_dir`` (sorted)."""
        markdown_files = [
            os.path.join(self.files_dir, fn) for fn in os.listdir(self.files_dir) if fn.endswith(".md")
        ]
        logger.info(f"Found {len(markdown_files)} markdown files in {self.files_dir}")
        return sorted(markdown_files)

    async def _load_all_documents(self) -> list[Document]:
        """Load and cache every document in the corpus."""
        if self.documents_cache is not None:
            return self.documents_cache

        markdown_files = self._get_all_markdown_files()
        documents = []

        for file_path in tqdm(markdown_files, desc="Loading documents"):
            filename = os.path.basename(file_path)
            celeb_name = filename.replace(".md", "")

            try:
                document = await Document.from_markdown(
                    file_path,
                    description=self.description,
                    document_name=celeb_name,
                    **self.config.get("document_config", {}),
                )
                documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to load document {file_path}: {e}")

        logger.info(f"Successfully loaded {len(documents)} documents")
        self.documents_cache = documents
        return documents

    @property
    def description(self) -> str:
        return "Wikipedia biography of a celebrity or artist"

    async def _run_row(self, row: dict, system: System, all_metadata: list) -> dict:
        question = row["question"]
        question_id = row["id"]

        all_documents = await self._load_all_documents()
        logger.info(f"Processing question with {len(all_documents)} documents")

        if self.config.get("docprocessing", True):
            try:
                processed_documents = await contextualize_document_metadata(all_documents, question, model=self.config.get("document_config", {}).get("description_model", "gpt-4.1-mini"))
            except Exception as e:
                logger.warning(f"Error contextualizing documents: {e}, using original documents")
                processed_documents = all_documents
        else:
            processed_documents = all_documents

        try:
            answer, metadata = await system.run(question, processed_documents, question_id=question_id)
            metadata["predicted_answer"] = answer
            metadata["question"] = question
        except Exception as e:
            logger.error(f"Error running system for question: {question}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error(e)
            error_metadata = {
                "question": question,
                "error": str(e),
                "answer": None,
                "metadata": None,
                "question_id": question_id,
            }
            all_metadata.append(error_metadata)
            return {"error": str(e), "question_id": question_id, "question": question}

        metadata["id"] = question_id

        result = {
            "question_id": question_id,
            "question": question,
            "predicted_answer": answer,
        }

        all_metadata.append(metadata)
        return result

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
        **kwargs,
    ) -> dict:
        import random

        results = []
        questions = self.questions.copy()

        if filter_func is not None:
            questions = [q for q in questions if filter_func(q)]

        if sample_size is not None:
            if random_state is not None:
                random.seed(random_state)
            questions = random.sample(questions, min(sample_size, len(questions)))

        all_metadata = []
        dataset_size = len(questions)

        logger.info(f"Running SLIDERS on {dataset_size} question(s) over the provided corpus")
        await self._load_all_documents()

        if parallel:
            tasks = [self._run_row(row, system, all_metadata) for row in questions]
            results = await tqdm_asyncio.gather(*tasks, desc="Processing questions")
        else:
            for idx, row in enumerate(tqdm(questions, desc="Running experiment")):
                logger.info(
                    f"===============================================\n"
                    f"{idx + 1} of {dataset_size} | Question {row['id']}\n"
                    f"==============================================="
                )
                logger.info(f"Question: {row['question']}")

                result = await self._run_row(row, system, all_metadata)
                results.append(result)
                log_wiki_celeb_results(result)

                if "error" in result:
                    logger.warning(f"Question {row['id']} had an error: {result['error']}")
                logger.info(f"Completed {len(results)}/{dataset_size} questions")

        successful_count = len([m for m in all_metadata if "error" not in m])
        error_count = len([m for m in all_metadata if "error" in m])
        logger.info("=== SLIDERS RUN COMPLETE ===")
        logger.info(f"Total questions processed: {len(results)}")
        logger.info(f"Successful runs: {successful_count}")
        logger.info(f"Errors: {error_count}")

        return {
            "experiment_id": SlidersGlobal.experiment_id,
            "results": results,
            "all_metadata": all_metadata,
        }
