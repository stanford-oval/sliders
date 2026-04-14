from langchain_text_splitters import RecursiveCharacterTextSplitter

from sliders.document import Document
from sliders.llm_models import ManagerResponse, SequentialAnswer, WorkerResponse
from sliders.llm_tools.code import run_python_code
from sliders.log_utils import logger
from sliders.system import System
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler
from langgraph.prebuilt import create_react_agent

# Optional RLM import
try:
    from rlm import RLM
    from rlm.logger import RLMLogger

    RLM_AVAILABLE = True
    rlm_logger = RLMLogger(log_dir="rlm_logs")
except ImportError:
    RLM_AVAILABLE = False
    RLM = None
    RLMLogger = None
    rlm_logger = None
    logger.warning("RLM package not installed. RLMSystem will not be available.")


class LLMWithToolUseSystem(System):
    def _setup_chains(self):
        tool_use_llm_client = get_llm_client(**self.config["models"]["tool_use"])
        self.tool_use_agent = create_react_agent(
            model=tool_use_llm_client,
            tools=[run_python_code],
        )

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        metadata = {}
        logger.info(f"Running tool use system for question: {question}")

        handler = LoggingHandler(
            prompt_file="default",
            metadata={
                "question": question,
                "stage": "tool_use",
                "question_id": kwargs.get("question_id", None),
            },
        )

        documents = "\n".join([document.content for document in documents])
        res = await self.tool_use_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Question: {question}
# Documents 
{documents}
""",
                    }
                ]
            },
            config={"callbacks": [handler]},
        )

        return res["messages"][-1].content, metadata


class LLMWithoutToolUseSystem(System):
    def _setup_chains(self):
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file="baselines/direct_without_tool_use.prompt",
            template_blocks=[],
        )
        self.answer_chain = answer_template | answer_llm_client

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        logger.info(f"Running without tool use system for question: {question}")
        metadata = {}
        handler = LoggingHandler(
            prompt_file="baselines/direct_without_tool_use.prompt",
            metadata={
                "question": question,
                "stage": "answer",
                "question_id": kwargs.get("question_id", None),
                **(metadata or {}),
            },
        )
        res = await self.answer_chain.ainvoke(
            {"question": question, "document": "\n".join([document.content for document in documents])},
            config={"callbacks": [handler]},
        )
        res = res.content
        metadata["answer_chain"] = str(res)
        return res, metadata


class LLMSequentialSystem(System):
    def _setup_chains(self):
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file=self.config["models"]["answer"]["template_file"],
            template_blocks=[],
        )
        self.answer_chain = answer_template | answer_llm_client.with_structured_output(SequentialAnswer)

    async def run(self, question: str, document: Document, *args, **kwargs) -> str:
        for i, chunk in enumerate(document.chunks):
            handler = LoggingHandler(
                prompt_file=self.config["models"]["answer"]["template_file"],
                metadata={
                    "question": question,
                    "stage": "answer",
                    "question_id": kwargs.get("question_id", None),
                },
            )
            res = await self.answer_chain.ainvoke(
                {
                    "question": question,
                    "document": f"# Chunk ({i}/{len(document.chunks)})\n\n" + chunk["content"],
                    "last_scratchpad": "",
                },
                config={"callbacks": [handler]},
            )
            if res.found_answer:
                logger.info(f"Found answer in chunk {i}, returning answer")
                break
        return res


class QuestionGuidedBaselineSystem(System):
    """Baseline that gathers question-focused notes per chunk and answers directly from them."""

    EXTRACTION_PROMPT = "baselines/question_guided_extract.prompt"
    ANSWER_PROMPT = "baselines/question_guided_answer.prompt"

    def _setup_chains(self):
        models_cfg = self.config.get("models", {})
        extract_cfg = models_cfg.get("extract")
        answer_cfg = models_cfg.get("answer")

        if extract_cfg is None or answer_cfg is None:
            raise ValueError("QuestionGuidedBaselineSystem expects 'extract' and 'answer' model configs.")

        extract_llm_client = get_llm_client(**extract_cfg)
        extract_template = load_fewshot_prompt_template(
            template_file=self.EXTRACTION_PROMPT,
            template_blocks=[],
        )
        self.extract_chain = extract_template | extract_llm_client

        answer_llm_client = get_llm_client(**answer_cfg)
        answer_template = load_fewshot_prompt_template(
            template_file=self.ANSWER_PROMPT,
            template_blocks=[],
        )
        self.answer_chain = answer_template | answer_llm_client

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> tuple[str, dict]:
        logger.info(f"Running question-guided baseline for question: {question}")
        question_id = kwargs.get("question_id", None)

        metadata: dict = {
            "question": question,
            "question_id": question_id,
            "extractions": [],
            "num_documents": len(documents),
        }

        aggregated_snippets: list[str] = []

        for doc_idx, document in enumerate(documents):
            for chunk_idx, chunk in enumerate(document.chunks):
                chunk_content = chunk.get("content", "").strip()
                if not chunk_content:
                    continue

                handler = LoggingHandler(
                    prompt_file=self.EXTRACTION_PROMPT,
                    metadata={
                        "question": question,
                        "stage": "extract",
                        "question_id": question_id,
                        "document_index": doc_idx,
                        "document_name": document.document_name,
                        "chunk_index": chunk_idx,
                    },
                )
                res = await self.extract_chain.ainvoke(
                    {
                        "question": question,
                        "chunk": chunk_content,
                        "document_name": document.document_name,
                    },
                    config={"callbacks": [handler]},
                )

                extract_text = res.content if hasattr(res, "content") else str(res)
                extract_text = (extract_text or "").strip()

                if not extract_text or extract_text.upper() == "NO_RELEVANT_INFORMATION":
                    continue

                snippet = f"{document.document_name} (chunk {chunk_idx}): {extract_text}"
                aggregated_snippets.append(snippet)
                metadata["extractions"].append(
                    {
                        "document": document.document_name,
                        "chunk_index": chunk_idx,
                        "text": extract_text,
                        "chunk_metadata": chunk.get("metadata"),
                    }
                )

        aggregated_context = "\n\n".join(aggregated_snippets)
        metadata["aggregated_snippets"] = len(aggregated_snippets)
        metadata["aggregated_context_chars"] = len(aggregated_context)

        if not aggregated_context:
            logger.info("No relevant extractions found; falling back to full document text.")
            metadata["fallback_to_full_text"] = True
            aggregated_context = "\n\n".join(document.content for document in documents if document.content.strip())
        else:
            metadata["fallback_to_full_text"] = False

        handler = LoggingHandler(
            prompt_file=self.ANSWER_PROMPT,
            metadata={
                "question": question,
                "stage": "answer",
                "question_id": question_id,
            },
        )
        answer_res = await self.answer_chain.ainvoke(
            {
                "question": question,
                "extracted_context": aggregated_context,
            },
            config={"callbacks": [handler]},
        )
        answer_text = answer_res.content if hasattr(answer_res, "content") else str(answer_res)
        answer_text = (answer_text or "").strip()
        metadata["answer"] = answer_text

        return answer_text, metadata


class ChainOfAgentsSystem(System):
    """Chain-of-Agents (CoA) system per arxiv 2406.02818.
    Stage 1 — Workers: sequential agents each read one chunk + the previous agent's
    communication unit (CU), then pass an updated CU forward.
    Stage 2 — Manager: receives only the final CU and produces the answer."""

    WORKER_PROMPT = "baselines/coa_worker.prompt"
    MANAGER_PROMPT = "baselines/coa_manager.prompt"

    def _setup_chains(self):
        models_cfg = self.config.get("models", {})
        worker_cfg = models_cfg.get("worker")
        manager_cfg = models_cfg.get("manager")

        if worker_cfg is None or manager_cfg is None:
            raise ValueError("ChainOfAgentsSystem expects 'worker' and 'manager' model configs.")

        worker_llm = get_llm_client(**worker_cfg)
        worker_template = load_fewshot_prompt_template(
            template_file=self.WORKER_PROMPT,
            template_blocks=[],
        )
        self.worker_chain = worker_template | worker_llm.with_structured_output(WorkerResponse)

        manager_llm = get_llm_client(**manager_cfg)
        manager_template = load_fewshot_prompt_template(
            template_file=self.MANAGER_PROMPT,
            template_blocks=[],
        )
        self.manager_chain = manager_template | manager_llm.with_structured_output(ManagerResponse)

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> tuple[str, dict]:
        logger.info(f"Running Chain-of-Agents system for question: {question}")
        question_id = kwargs.get("question_id", None)

        # Plain chunking of raw document text (no table handling)
        chunk_size = self.config.get("chunk_size", 16000)
        chunk_overlap = self.config.get("chunk_overlap", 0)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for document in documents:
            texts = splitter.split_text(document.content)
            for text in texts:
                all_chunks.append({"content": text, "document_name": document.document_name})

        total_chunks = len(all_chunks)
        metadata = {
            "question": question,
            "question_id": question_id,
            "num_documents": len(documents),
            "total_chunks": total_chunks,
            "worker_evidence": [],
        }

        # Stage 1 — Workers: sequential processing, each passes CU to next
        communication = "No previous evidence yet."
        for i, chunk_info in enumerate(all_chunks):
            chunk_label = f"Chunk ({i + 1}/{total_chunks}) from {chunk_info['document_name']}"
            logger.info(f"Worker processing {chunk_label}")

            handler = LoggingHandler(
                prompt_file=self.WORKER_PROMPT,
                metadata={
                    "question": question,
                    "stage": "worker",
                    "question_id": question_id,
                    "chunk_index": i,
                    "document_name": chunk_info["document_name"],
                },
            )

            worker_result = await self.worker_chain.ainvoke(
                {
                    "question": question,
                    "previous_communication": communication,
                    "chunk_label": chunk_label,
                    "chunk": chunk_info["content"],
                },
                config={"callbacks": [handler]},
            )

            communication = worker_result.communication
            metadata["worker_evidence"].append(
                {
                    "chunk_index": i,
                    "document_name": chunk_info["document_name"],
                    "evidence": worker_result.evidence,
                }
            )

        # Stage 2 — Manager: synthesize final answer from last CU
        logger.info("Manager synthesizing final answer")

        handler = LoggingHandler(
            prompt_file=self.MANAGER_PROMPT,
            metadata={
                "question": question,
                "stage": "manager",
                "question_id": question_id,
            },
        )

        manager_result = await self.manager_chain.ainvoke(
            {
                "question": question,
                "accumulated_evidence": communication,
            },
            config={"callbacks": [handler]},
        )

        answer = manager_result.answer
        metadata["answer"] = answer
        metadata["manager_reasoning"] = manager_result.reasoning

        return answer, metadata


class RLMSystem(System):
    def _setup_chains(self):
        if not RLM_AVAILABLE:
            raise ImportError(
                "RLM package is not installed. Please install it to use RLMSystem. "
                "You can install it with: pip install rlm"
            )

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        if not RLM_AVAILABLE:
            raise ImportError(
                "RLM package is not installed. Please install it to use RLMSystem. "
                "You can install it with: pip install rlm"
            )

        models_config = self.config.get("models", {})
        main_model = models_config.get("main", {}).get("model", "gpt-5")
        sub_model = models_config.get("sub", {}).get("model", "gpt-5-mini")
        max_iterations = self.config.get("max_iterations", 30)

        metadata = {"question": question}
        rlm = RLM(
            backend="azure_openai",
            backend_kwargs={"model_name": main_model},
            other_backends=["azure_openai"],
            other_backend_kwargs=[{"model_name": sub_model}],
            verbose=True,
            logger=rlm_logger,
            max_iterations=max_iterations,
        )
        context = "\n".join([document.content for document in documents])
        res = rlm.completion(prompt=context, root_prompt=question).response
        return res, metadata
