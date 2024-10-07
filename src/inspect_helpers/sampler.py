import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from inspect_ai.log import EvalLog, EvalSample
from pydantic import BaseModel
import warnings

class EvalSampler:
    def __init__(self, eval_log: EvalLog):
        self.eval_log = eval_log
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.eval_log.samples:
            warnings.warn("No samples found in eval_log. Returning an empty DataFrame.")
            return pd.DataFrame()

        data = []
        for sample in self.eval_log.samples:
            sample_dict = sample.dict()
            flattened = self._flatten_dict(sample_dict)
            data.append(flattened)
        
        return pd.DataFrame(data)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def filter_samples(self, conditions: List[Tuple[str, Any]]) -> pd.DataFrame:
        filtered_df = self.df.copy()
        for column, value in conditions:
            matching_columns = filtered_df.filter(like=column).columns
            if len(matching_columns) == 1:
                filtered_df = filtered_df[filtered_df[matching_columns[0]] == value]
            elif len(matching_columns) > 1:
                filtered_df = filtered_df[filtered_df[matching_columns].eq(value).any(axis=1)]
        return filtered_df

    def rank_samples(
        self,
        rank_column: str,
        n: int = 10,
        ascending: bool = False,
        conditions: Optional[List[Tuple[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Filters and ranks samples based on specified conditions and a ranking column.

        Args:
            rank_column (str): The column name to rank by.
            n (int, optional): Number of top/bottom samples to retrieve. Defaults to 10.
            ascending (bool, optional): Sort order. False for descending, True for ascending. Defaults to False.
            conditions (List[Tuple[str, Any]], optional): Filtering conditions as a list of (column, value) tuples. Defaults to None.

        Returns:
            pd.DataFrame: The filtered and ranked DataFrame.
        """
        if conditions:
            filtered_df = self.filter_samples(conditions)
        else:
            filtered_df = self.df.copy()

        matching_columns = filtered_df.filter(like=rank_column).columns
        if len(matching_columns) != 1:
            warnings.warn(f"Multiple or no columns match '{rank_column}'. Please specify a more precise column name.")
            return pd.DataFrame()
        if ascending:
            return filtered_df.nsmallest(n, matching_columns[0])
        else:
            return filtered_df.nlargest(n, matching_columns[0])