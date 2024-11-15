import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from inspect_ai.log import EvalLog, EvalSample
from pydantic import BaseModel
import warnings
import operator

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
            sample_dict = sample.model_dump()
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

    def filter_samples(self, conditions: List[Union[Tuple[str, Any], Tuple[str, str, Any]]]) -> pd.DataFrame:
        filtered_df = self.df.copy()
        for condition in conditions:
            if len(condition) == 2:
                column, value = condition
                op = '=='
            elif len(condition) == 3:
                column, op, value = condition
            else:
                raise ValueError("Invalid condition format. Use (column, value) or (column, operator, value).")

            matching_columns = filtered_df.filter(like=column).columns
            if len(matching_columns) == 1:
                filtered_df = filtered_df[self._apply_condition(filtered_df[matching_columns[0]], op, value)]
            elif len(matching_columns) > 1:
                filtered_df = filtered_df[filtered_df[matching_columns].apply(lambda col: self._apply_condition(col, op, value)).any(axis=1)]
        return filtered_df

    def _apply_condition(self, series: pd.Series, op: str, value: Any) -> pd.Series:
        ops = {
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            'in': lambda x, y: x.isin(y) if isinstance(y, (list, tuple)) else x == y,
            'not in': lambda x, y: ~x.isin(y) if isinstance(y, (list, tuple)) else x != y,
            'between': lambda x, y: (x >= y[0]) & (x <= y[1])
        }
        op = op.lower()
        if op == 'between' and not isinstance(value, (list, tuple)) or (isinstance(value, (list, tuple)) and len(value) != 2):
            raise ValueError("'between' operation requires a list or tuple of two values.")
        return ops[op](series, value)

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
