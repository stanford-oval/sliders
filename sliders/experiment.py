import datetime
import os
import numpy as np
from sliders.log_utils import logger
import matplotlib.pyplot as plt


def print_result_summary(results: list[dict]):
    timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_save_path = "results/summaries/"
    os.makedirs(base_save_path, exist_ok=True)

    summary_text_file = SummaryTextFile(os.path.join(base_save_path, f"results_summary_{timestamp_suffix}.txt"))

    if isinstance(results, dict) and "results_summary" in results:
        logger.info("Experiment Summary already generated, printing...")
        for tool_name, tool_data in results["results_summary"].items():
            if "accuracy" in tool_data:
                summary_text_file.write(f"{tool_name} | {tool_data['accuracy']:.2f} (N={tool_data['total']})\n")
        return

    if len(results) != 2:
        summary_text_file.write("Experiment does not return results and metadata, skipping summary")
        return
    result, metadata = results
    if len(result) == 0:
        summary_text_file.write("No results found, skipping summary")
        return

    eval_tool_names = list(result[0]["evaluation_tools"].keys())
    eval_tool_names.sort()
    eval_tool_results = {tool: [] for tool in eval_tool_names}

    summary_text_file.write("\nEXPERIMENT SUMMARY: ========================================")

    for i, question_result in enumerate(result):
        for tool in eval_tool_names:
            if tool not in question_result["evaluation_tools"]:
                logger.info(f"Evaluation {tool} not found for question {i}")
                continue
            if "error" in question_result["evaluation_tools"][tool]:
                logger.info(
                    f"Error for question {i} and tool {tool}: {question_result['evaluation_tools'][tool]['error']}"
                )
                score = 0
            else:
                score = int(question_result["evaluation_tools"][tool]["correct"])
            eval_tool_results[tool].append(score)

    summary_text_file.write(f"\nShowing result summary for {len(eval_tool_results[eval_tool_names[0]])} questions")
    for tool in eval_tool_names:
        avg_score = np.mean(eval_tool_results[tool])
        std_score = np.std(eval_tool_results[tool])
        confidence_interval = 1.96 * std_score / np.sqrt(len(eval_tool_results[tool]))
        summary_text_file.write(f"{tool} | {avg_score:.2f} ± {confidence_interval:.2f}")

        if tool == "LLMAsJudgeEvaluationToolloong_evaluator":
            avg_score = np.mean(np.array(eval_tool_results[tool]) == 100)
            std_score = np.std(np.array(eval_tool_results[tool]) == 100)
            confidence_interval = 1.96 * std_score / np.sqrt(len(eval_tool_results[tool]))
            summary_text_file.write(f"{tool}_perf_rate | {avg_score:.2f} ± {confidence_interval:.2f}")

    if "misc_question_metadata" in metadata[0].keys():
        summary_text_file.write("\nQuestion Type Breakdown: -----------------------------------")
        question_metadata_names = list(metadata[0]["misc_question_metadata"].keys())

        question_type_metadata = {name: [] for name in question_metadata_names}
        for i, question_metadata in enumerate(metadata):
            for metadata_name in question_metadata_names:
                if metadata_name not in question_metadata["misc_question_metadata"]:
                    logger.info(f"Metadata {metadata_name} not found for question {i}")
                    continue
                question_type_metadata[metadata_name].append(question_metadata["misc_question_metadata"][metadata_name])

        for metadata_name in question_metadata_names:
            summary_text_file.write(f"--- {metadata_name}: ---------------------------------------")
            # mask the metadata to the results for each unique value and give the mean and std for each tool/category
            unique_values = list(set(question_type_metadata[metadata_name]))
            unique_values.sort()
            for value in unique_values:
                mask = np.array(question_type_metadata[metadata_name]) == value
                summary_text_file.write(f"{value}: ------------------------------------  (N={mask.sum()})")
                for tool in eval_tool_names:
                    temp_eval_tool = np.asarray(eval_tool_results[tool])
                    avg_score = np.mean(temp_eval_tool[mask])
                    std_score = np.std(temp_eval_tool[mask])
                    confidence_interval = 1.96 * std_score / np.sqrt(mask.sum())
                    summary_text_file.write(f"    {tool}: {avg_score:.2f} ± {confidence_interval:.2f}")
                    if tool == "LLMAsJudgeEvaluationToolloong_evaluator":
                        avg_score = np.mean(np.array(temp_eval_tool[mask]) == 100)
                        std_score = np.std(np.array(temp_eval_tool[mask]) == 100)
                        confidence_interval = 1.96 * std_score / np.sqrt(mask.sum())
                        summary_text_file.write(f"    {tool}_perf_rate: {avg_score:.2f} ± {confidence_interval:.2f}")

        if len(question_metadata_names) == 2:
            # make a heatmap based on the mean
            # sort the question_metadata_names by the number of unique values in descending order
            question_metadata_names.sort(key=lambda x: len(set(question_type_metadata[x])), reverse=True)
            metadata_name_x, metadata_name_y = question_metadata_names
            heatmap_save_paths = generate_results_heatmap(
                metadata_name_x,
                metadata_name_y,
                eval_tool_names,
                eval_tool_results,
                question_type_metadata,
                timestamp_suffix,
            )
            for tool_name, heatmap_save_path in heatmap_save_paths.items():
                summary_text_file.write(f"Saved {tool_name} heatmap to {heatmap_save_path}")
            logger.info(f"Saved results summary to {summary_text_file.file_path}")

    summary_text_file.close()


def generate_results_heatmap(
    metadata_name_x: str,
    metadata_name_y: str,
    eval_tool_names: list[str],
    eval_tool_results: dict,
    q_metadata: dict,
    timestamp_suffix: str,
    base_save_path: str = "results/heatmaps/",
):
    save_paths = {}
    # get the unique values for each metadata
    unique_values_x = list(set(q_metadata[metadata_name_x]))
    unique_values_x.sort()
    unique_values_y = list(set(q_metadata[metadata_name_y]))
    unique_values_y.sort()

    # convert metadata and results to numpy arrays for masking/indexing
    arr_x = np.array(q_metadata[metadata_name_x])
    arr_y = np.array(q_metadata[metadata_name_y])
    tool_results_np = {tool: np.array(eval_tool_results[tool]) for tool in eval_tool_names}

    # create a heatmap based on the mean of the eval_tool_results for each unique value of each metadata
    heatmap = np.zeros((len(eval_tool_names), len(unique_values_y), len(unique_values_x)))
    sample_counts = np.zeros((len(unique_values_y), len(unique_values_x)), dtype=int)

    # Calculate aggregated accuracy for each metadata value to use as tick labels
    x_metadata_accuracy = {}
    y_metadata_accuracy = {}

    for value_x in unique_values_x:
        mask_x = arr_x == value_x
        x_metadata_accuracy[value_x] = {}
        for tool_name in eval_tool_names:
            if mask_x.sum() > 0:
                x_metadata_accuracy[value_x][tool_name] = np.mean(tool_results_np[tool_name][mask_x])
            else:
                x_metadata_accuracy[value_x][tool_name] = 0.0

    for value_y in unique_values_y:
        mask_y = arr_y == value_y
        y_metadata_accuracy[value_y] = {}
        for tool_name in eval_tool_names:
            if mask_y.sum() > 0:
                y_metadata_accuracy[value_y][tool_name] = np.mean(tool_results_np[tool_name][mask_y])
            else:
                y_metadata_accuracy[value_y][tool_name] = 0.0

    for i, value_y in enumerate(unique_values_y):
        for j, value_x in enumerate(unique_values_x):
            mask = (arr_y == value_y) & (arr_x == value_x)
            sample_counts[i, j] = int(mask.sum())
            for tool_idx, tool_name in enumerate(eval_tool_names):
                if sample_counts[i, j] > 0:
                    heatmap[tool_idx, i, j] = np.mean(tool_results_np[tool_name][mask])
                else:
                    heatmap[tool_idx, i, j] = np.nan

    # plot the heatmap
    for tool_idx, tool_name in enumerate(eval_tool_names):
        plt.imshow(heatmap[tool_idx], cmap="Blues", vmin=RESULT_RANGES[tool_name][0], vmax=RESULT_RANGES[tool_name][1])
        plt.colorbar()
        plt.title(f"{tool_name} N={len(eval_tool_results[tool_name])}")
        plt.xlabel(metadata_name_x)
        plt.ylabel(metadata_name_y)

        # Create tick labels showing aggregated accuracy for each metadata value
        x_tick_labels = [f"{value_x}\n({x_metadata_accuracy[value_x][tool_name]:.2f})" for value_x in unique_values_x]
        y_tick_labels = [f"{value_y}\n({y_metadata_accuracy[value_y][tool_name]:.2f})" for value_y in unique_values_y]

        # set tick labels to show the unique metadata values with aggregated accuracy
        plt.xticks(ticks=range(len(unique_values_x)), labels=x_tick_labels, rotation=45, ha="right")
        plt.yticks(ticks=range(len(unique_values_y)), labels=y_tick_labels)

        # annotate each cell with the sample size for that cell
        for i in range(len(unique_values_y)):
            for j in range(len(unique_values_x)):
                if not np.isnan(heatmap[tool_idx, i, j]):
                    plt.text(
                        j,
                        i,
                        f"{heatmap[tool_idx, i, j]:.2f}\nN={sample_counts[i, j]}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
                else:
                    plt.text(j, i, f"N/A\nN={sample_counts[i, j]}", ha="center", va="center", color="black", fontsize=8)

        os.makedirs(f"{base_save_path}/{tool_name}", exist_ok=True)
        plt.tight_layout()
        save_path = f"{base_save_path}/{tool_name}/score_heatmap_{timestamp_suffix}.png"
        plt.savefig(save_path)
        plt.close()
        save_paths[tool_name] = save_path

    return save_paths


RESULT_RANGES = {
    "LLMAsJudgeEvaluationToolsoft_evaluator": (0, 1),
    "LLMAsJudgeEvaluationToolsoft_evaluator_with_ranges": (0, 1),
    "LLMAsJudgeEvaluationToolhard_evaluator": (0, 1),
    "LLMAsJudgeEvaluationToolloong_evaluator": (1, 100),
}


class SummaryTextFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = open(file_path, "a")

    def write(self, text: str):
        self.file.write("\n" + text)
        logger.info(text)

    def close(self):
        self.file.close()
