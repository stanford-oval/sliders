import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class Dataset:
    """A class for loading and iterating over datasets from various file formats."""

    def __init__(self, path: str):
        """Initialize the dataset with a file path.

        Args:
            path: Path to the dataset file (supports .jsonl, .json, .csv)
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.data = self._load_data()

    def _load_data(self) -> list:
        """Load data from the file based on its extension."""
        if self.path.suffix == ".jsonl":
            return self._load_jsonl()
        elif self.path.suffix == ".json":
            return self._load_json()
        elif self.path.suffix == ".csv":
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {self.path.suffix}")

    def _load_jsonl(self) -> list:
        """Load data from a JSONL file."""
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_json(self) -> list:
        """Load data from a JSON file."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If the JSON contains a single object, wrap it in a list
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("JSON file must contain either a list or a single object")

    def _load_csv(self) -> list:
        """Load data from a CSV file."""
        try:
            import pandas as pd

            df = pd.read_csv(self.path)
            return df.to_dict("records")
        except ImportError:
            raise ImportError("pandas is required to load CSV files")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make the dataset iterable."""
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get an item by index."""
        return self.data[index]

    def filter(self, condition_func) -> "Dataset":
        """Filter the dataset based on a condition function.

        Args:
            condition_func: A function that takes a data item and returns True/False

        Returns:
            A new Dataset instance with filtered data
        """
        filtered_data = [item for item in self.data if condition_func(item)]
        new_dataset = Dataset.__new__(Dataset)
        new_dataset.path = self.path
        new_dataset.data = filtered_data
        return new_dataset

    def sample(self, n: int, random_state: Optional[int] = None) -> "Dataset":
        """Sample n items from the dataset.

        Args:
            n: Number of items to sample
            random_state: Random seed for reproducibility

        Returns:
            A new Dataset instance with sampled data
        """
        import random

        if random_state is not None:
            random.seed(random_state)

        sampled_data = random.sample(self.data, min(n, len(self.data)))
        new_dataset = Dataset.__new__(Dataset)
        new_dataset.path = self.path
        new_dataset.data = sampled_data
        return new_dataset

    def filter_by_specific_ids(self, ids: list[str]) -> "Dataset":
        """Filter the dataset based on specific IDs from a CSV file."""
        specific_ids = {str(item) for item in ids}
        new_dataset = Dataset.__new__(Dataset)
        new_dataset.path = self.path
        new_dataset.data = [
            item for item in self.data if (item_id := item.get("id")) is not None and str(item_id) in specific_ids
        ]
        return new_dataset
