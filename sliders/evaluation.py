from abc import ABC, abstractmethod
import asyncio
import os
from typing import Any, Dict, List

from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from pydantic import BaseModel

from sliders.llm_models import Evaluation
from sliders.callbacks.logging import LoggingHandler


class EvaluationTool(ABC):
    """Abstract base class for evaluation tools."""

    @abstractmethod
    async def evaluate(
        self, gold_answer: str, predicted_answer: str, question: str, question_id: str, tool_name: str, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a predicted answer against a gold answer.

        Args:
            gold_answer: The ground truth answer
            predicted_answer: The predicted answer to evaluate
            question: The original question
            question_id: The id of the question
            tool_name: The name of the evaluation tool
            **kwargs: Additional arguments specific to the evaluation tool

        Returns:
            Dict containing evaluation results
        """
        pass


class Evaluator:
    """Main evaluator class that manages multiple evaluation tools."""

    def __init__(self, evaluation_tools: List[EvaluationTool] = None):
        """Initialize the evaluator with a list of evaluation tools.

        Args:
            evaluation_tools: List of evaluation tools to use
        """
        self.evaluation_tools = evaluation_tools or []

    def add_evaluation_tool(self, tool: EvaluationTool):
        """Add an evaluation tool to the evaluator.

        Args:
            tool: The evaluation tool to add
        """
        self.evaluation_tools.append(tool)

    def remove_evaluation_tool(self, tool: EvaluationTool):
        """Remove an evaluation tool from the evaluator.

        Args:
            tool: The evaluation tool to remove
        """
        if tool in self.evaluation_tools:
            self.evaluation_tools.remove(tool)

    async def evaluate(
        self, question_id, gold_answer: str, predicted_answer: str, question: str, **kwargs
    ) -> Dict[str, Any]:
        """Run evaluation using all configured evaluation tools.

        Args:
            gold_answer: The ground truth answer
            predicted_answer: The predicted answer to evaluate
            question: The original question
            **kwargs: Additional arguments to pass to evaluation tools

        Returns:
            Dict containing results from all evaluation tools
        """
        results = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "evaluation_tools": {},
        }

        all_tool_tasks = []
        for i, tool in enumerate(self.evaluation_tools):
            tool_name = tool.__class__.__name__
            # If multiple tools of the same type, add index
            if tool_name in results:
                tool_name = f"{tool_name}_{i}"

            if tool_name == "LLMAsJudgeEvaluationTool":
                tool_name = "LLMAsJudgeEvaluationTool" + tool.prompt_file.split("/")[-1].split(".")[0]
            all_tool_tasks.append(
                tool.evaluate(gold_answer, predicted_answer, question, question_id, tool_name, **kwargs)
            )

        all_tool_results = await asyncio.gather(*all_tool_tasks)

        for tool, tool_result in zip(self.evaluation_tools, all_tool_results):
            tool_name = tool.__class__.__name__

            if tool_name == "LLMAsJudgeEvaluationTool":
                tool_name = "LLMAsJudgeEvaluationTool_" + tool.prompt_file.split("/")[-1].split(".")[0]
            if "error" in tool_result:
                results["evaluation_tools"][tool_name] = {"error": tool_result["error"]}
            else:
                results["evaluation_tools"][tool_name] = tool_result

        return results


class LLMAsJudgeEvaluationTool(EvaluationTool):
    """Evaluation tool that uses an LLM as a judge to evaluate answers."""

    def __init__(
        self,
        prompt_file: str,
        eval_class: BaseModel = Evaluation,
        model: str = "gpt-4.1",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        llm_config: dict | None = None,
    ):
        """Initialize the LLM as judge evaluation tool.

        Args:
            prompt_file: Path to the prompt template file
            eval_class: Type of evaluation to perform
            model: The LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.prompt_file = prompt_file
        self.eval_class = eval_class
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_config = dict(llm_config or {})
        self.chain = self._create_chain()

    def _create_chain(self):
        """Create the LLM generation chain."""
        llm_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.llm_config,
        }

        # Global evaluator-only env overrides (no YAML config changes required).
        # These apply only to evaluation LLM calls and let users route evaluators to
        # a separate deployment/endpoint from the main pipeline.
        eval_provider = os.getenv("EVAL_LLM_PROVIDER")
        if eval_provider and "provider" not in llm_kwargs:
            llm_kwargs["provider"] = eval_provider

        eval_openai_base_url = os.getenv("EVAL_OPENAI_BASE_URL")
        if eval_openai_base_url and "base_url" not in llm_kwargs:
            llm_kwargs["base_url"] = eval_openai_base_url

        eval_openai_api_key = os.getenv("EVAL_OPENAI_API_KEY")
        if eval_openai_api_key and "api_key" not in llm_kwargs:
            llm_kwargs["api_key"] = eval_openai_api_key

        eval_azure_endpoint = os.getenv("EVAL_AZURE_OPENAI_ENDPOINT")
        if eval_azure_endpoint and "azure_endpoint" not in llm_kwargs:
            llm_kwargs["azure_endpoint"] = eval_azure_endpoint

        eval_azure_api_key = os.getenv("EVAL_AZURE_OPENAI_API_KEY")
        if eval_azure_api_key and "api_key" not in llm_kwargs:
            llm_kwargs["api_key"] = eval_azure_api_key

        eval_azure_api_version = os.getenv("EVAL_AZURE_OPENAI_API_VERSION")
        if eval_azure_api_version and "api_version" not in llm_kwargs:
            llm_kwargs["api_version"] = eval_azure_api_version

        # Optional indirection to avoid hardcoding secrets/URLs in config files.
        api_key_env_var = llm_kwargs.pop("api_key_env_var", None)
        if api_key_env_var and "api_key" not in llm_kwargs:
            env_key = os.getenv(api_key_env_var)
            if env_key:
                llm_kwargs["api_key"] = env_key

        base_url_env_var = llm_kwargs.pop("base_url_env_var", None)
        if base_url_env_var and "base_url" not in llm_kwargs:
            env_base_url = os.getenv(base_url_env_var)
            if env_base_url:
                llm_kwargs["base_url"] = env_base_url

        azure_endpoint_env_var = llm_kwargs.pop("azure_endpoint_env_var", None)
        if azure_endpoint_env_var and "azure_endpoint" not in llm_kwargs:
            env_azure_endpoint = os.getenv(azure_endpoint_env_var)
            if env_azure_endpoint:
                llm_kwargs["azure_endpoint"] = env_azure_endpoint

        llm_client = get_llm_client(**llm_kwargs)
        prompt_template = load_fewshot_prompt_template(template_file=self.prompt_file, template_blocks=[])
        return prompt_template | llm_client.with_structured_output(self.eval_class)

    async def evaluate(
        self, gold_answer: str, predicted_answer: str, question: str, question_id: str, tool_name: str, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate the predicted answer using LLM as judge.

        Args:
            gold_answer: The ground truth answer
            predicted_answer: The predicted answer to evaluate
            question: The original question
            question_id: The id of the question
            tool_name: The name of the evaluation tool
            **kwargs: Additional arguments

        Returns:
            Dict containing evaluation results
        """
        try:
            handler = LoggingHandler(
                prompt_file=self.prompt_file,
                metadata={
                    "question": question,
                    "stage": "evaluate",
                    "question_id": question_id,
                    "objective": tool_name,
                },
            )
            result = await self.chain.ainvoke(
                {"gold_answer": gold_answer, "predicted_answer": predicted_answer, "question": question},
                config={"callbacks": [handler]},
            )

            return {
                "explanation": result.explanation,
                "correct": result.correct,
                "model": self.model,
                "prompt_file": self.prompt_file,
                "experiment_id": None,
            }
        except Exception as e:
            return {
                "error": str(e),
                "model": self.model,
                "prompt_file": self.prompt_file,
                "experiment_id": None,
            }
