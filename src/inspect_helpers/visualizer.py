# %%
import pandas as pd
import altair as alt
from typing import List, Optional, Callable, Dict, Any
from inspect_ai.log import EvalLog, EvalLogInfo, read_eval_log
from inspect_ai.scorer import value_to_float


class EvalVisualizer:
    """
    A class for visualizing evaluation logs using Altair charts.

    Attributes:
        eval_logs (List[EvalLog]): List of evaluation logs to visualize.
        value_to_float_fn (Callable): Function to convert score values to floats.
        df (pd.DataFrame): DataFrame containing processed evaluation data.

    Methods:
        __init__(eval_logs, value_to_float_fn, model_categorizer): Initialize the visualizer.
        _create_dataframe(): Create a DataFrame from evaluation logs.
        _add_model_categories(model_categorizer): Add model categories to the DataFrame.
        visualize(...): Generate Altair charts based on the evaluation data.
    """
    def __init__(
        self,
        eval_log_infos: list[EvalLogInfo],
        value_to_float_fn: Callable[[Any], float] = value_to_float(),
        log_dir_categorizer: Optional[Callable[[str], Dict[str, str]]] = None,
        timestamp_categorizer: Optional[Callable[[float], Dict[str, str]]] = None,
        model_categorizer: Optional[Callable[[str], Dict[str, str]]] = None,
        rename_mappings: Dict[str, Dict[str, str]] = None,
        filter_sort_order: Dict[str, List[str]] = None,
    ):
        """
        Initialize the EvalVisualizer.

        Args:
            eval_logs (List[EvalLog]): List of evaluation logs.
            value_to_float_fn (Callable): Function to convert score values to floats.
            model_categorizer (Optional[Callable]): Function to further categorize models.
            rename_mappings (Dict[str, Dict[str, str]]): Mappings for renaming categories.
                Keys are column names, values are dictionaries mapping old names to new names.
            custom_sort_order (Dict[str, List[str]]): Custom sort order for categories.
                Keys are column names, values are lists defining the desired order using the new names.
        """
        self.eval_log_infos = eval_log_infos
        self.eval_logs = self._get_eval_logs()
        self.value_to_float_fn = value_to_float_fn
        self.filter_sort_order = filter_sort_order or {}
        self.rename_mappings = rename_mappings or {}
        self.df = self._create_dataframe()
        if log_dir_categorizer:
            self._add_log_dir_categories(log_dir_categorizer)
        if timestamp_categorizer:
            self._add_timestamp_categories(timestamp_categorizer)
        if model_categorizer:
            self._add_model_categories(model_categorizer)
        self._process_dataframe()


    def _get_eval_logs(self) -> list[EvalLog]:
        return [read_eval_log(eval_log_info) for eval_log_info in self.eval_log_infos]
    
    def _get_log_dir(self, name: str) -> str:
        return "/".join(name.split("///")[1].split("/")[:-1])

    def _create_dataframe(self) -> pd.DataFrame:
        data = []
        for i, log in enumerate(self.eval_logs):
            if log.status != "success" or not log.samples: #skip logs that are not successful or have no samples
                continue
            
            log_dir = self._get_log_dir(self.eval_log_infos[i].name)
            timestamp = self.eval_log_infos[i].mtime
            suffix = self.eval_log_infos[i].suffix
            run_id = log.eval.run_id
            task = log.eval.task
            model = log.eval.model
            dataset = log.eval.dataset.name
            
            for sample in log.samples:
                if sample.scores:
                    for scorer, score in sample.scores.items():
                        data.append(
                            {
                                "log_dir": log_dir,
                                "timestamp": timestamp,
                                "suffix": suffix,
                                "run_id": run_id,
                                "task": task,
                                "dataset": dataset,
                                "model": model,
                                "scorer": scorer,
                                "value": self.value_to_float_fn(score.value),
                            }
                        )
        return pd.DataFrame(data)
    
    def _add_log_dir_categories(self, log_dir_categorizer: Callable[[str], Dict[str, str]]):
        categories = self.df["log_dir"].apply(log_dir_categorizer)
        self.df = pd.concat([self.df, pd.DataFrame(categories.tolist())], axis=1)

    def _add_timestamp_categories(self, timestamp_categorizer: Callable[[float], Dict[str, str]]):
        categories = self.df["timestamp"].apply(timestamp_categorizer)
        self.df = pd.concat([self.df, pd.DataFrame(categories.tolist())], axis=1)

    def _add_model_categories(self, model_categorizer: Callable[[str], Dict[str, str]]):
        categories = self.df["model"].apply(model_categorizer)
        self.df = pd.concat([self.df, pd.DataFrame(categories.tolist())], axis=1)

    def _process_dataframe(self):
        # Apply rename mappings
        for column, mapping in self.rename_mappings.items():
            if column in self.df.columns:
                self.df[column] = self.df[column].map(mapping).fillna(self.df[column])
        
        # Apply custom sort order for filtering and sorting
        for column, order in self.filter_sort_order.items():
            if column in self.df.columns:
                # Create a categorical column with the custom sort order
                self.df[column] = pd.Categorical(self.df[column], categories=order, ordered=True)
                # Filter the dataframe to keep only the values in the custom sort order
                self.df = self.df[self.df[column].isin(order)]
        
        # Sort the dataframe based on all columns with custom sort order
        sort_columns = [col for col in self.filter_sort_order.keys() if col in self.df.columns]
        if sort_columns:
            self.df = self.df.sort_values(sort_columns)

    def visualize(
        self,
        fig_title: str = None,  # Add this parameter
        plot_fn: Callable[..., alt.Chart] = alt.Chart.mark_bar,
        plot_fn_kwargs: Dict[str, Any] = {},
        chart_properties: Dict[str, Any] = {},
        x_category: str = None,
        y_category: str = None,
        x_offset_category: str = None,
        y_offset_category: str = None,
        color_category: str = None,
        color_scheme: Optional[str] = None,
        color_domain: Optional[List[str]] = None,
        color_range: Optional[List[str]] = None,
        layer_category: str = None,
        facet_category: str = None,
        facet_columns: int = 3,
        h_concat_category: str = None,
        v_concat_category: str = None,
        tooltip_fields: List[alt.Tooltip] = None,
    ):
        """
        Generate Altair charts based on the evaluation data.

        Args:
            plot_fn (Callable): Altair chart type function.
            plot_fn_kwargs (Dict): Additional arguments for plot_fn.
            chart_properties (Dict): Properties for the chart.
            x_category (str): Category for x-axis.
            y_category (str): Category for y-axis.
            x_offset_category (str): Category for x-axis offset.
            y_offset_category (str): Category for y-axis offset.
            color_category (str): Category for color encoding.
            color_scheme (Optional[str]): Color scheme to use.
            color_domain (Optional[List[str]]): Color domain to use.
            color_range (Optional[List[str]]): Color range to use.
            layer_category (str): Category for layering charts.
            facet_category (str): Category for faceting charts.
            facet_columns (int): Number of columns in faceted charts.
            h_concat_category (str): Category for horizontal concatenation.
            v_concat_category (str): Category for vertical concatenation.
            rename_mappings (Dict): Mappings for renaming categories.
            fig_title (str): Title for the multi-chart figure.

        Returns:
            alt.Chart: Combined Altair chart.
        """
        charts = []
        df = self.df.copy()

        color_scale = alt.Scale()
        if color_scheme:
            color_scale = alt.Scale(scheme=color_scheme)
        if color_range:
            color_scale = alt.Scale(range=color_range)
        if color_domain:
            color_scale.domain = color_domain

        encoding = {
            "x": x_category,
            "y": y_category,
        }

        if tooltip_fields:
            encoding["tooltip"] = tooltip_fields

        base = alt.Chart(df).encode(**encoding)

        if color_category:
            base = base.encode(color=alt.Color(color_category, scale=color_scale))
        if x_offset_category:
            base = base.encode(xOffset=alt.XOffset(f"{x_offset_category}"))
        if y_offset_category:
            base = base.encode(yOffset=alt.YOffset(f"{y_offset_category}"))

        chart = plot_fn(base, **plot_fn_kwargs).properties(**chart_properties)

        if layer_category:
            chart = alt.layer(
                *[
                    chart.transform_filter(f"datum.{layer_category} == '{val}'")
                    for val in df[layer_category].unique()
                ]
            )

        if facet_category:
            chart = chart.facet(f"{facet_category}", columns=facet_columns)

        charts.append(chart)

        if h_concat_category:
            charts = [
                alt.hconcat(
                    *[
                        c.transform_filter(
                            f"datum.{h_concat_category} == '{val}'"
                        ).properties(title=f"{h_concat_category}: {val}")
                        for val in df[h_concat_category].unique()
                    ]
                )
                for c in charts
            ]

        if v_concat_category:
            charts = [
                alt.vconcat(
                    *[
                        c.transform_filter(
                            f"datum.{v_concat_category} == '{val}'"
                        ).properties(title=f"{v_concat_category}: {val}")
                        for val in df[v_concat_category].unique()
                    ]
                )
                for c in charts
            ]

        # Only create rows if there are charts to combine
        if len(charts) > 1:
            rows = [alt.hconcat(*charts)]
            combined_chart = alt.vconcat(*rows)
        else:
            combined_chart = charts[0]

        # Add the centered fig_title if provided
        if fig_title:
            combined_chart = combined_chart.properties(
                title=alt.TitleParams(
                    text=fig_title,
                    align="center",
                    anchor="middle"
                )
            )

        return combined_chart