# %%
import pandas as pd
import altair as alt
from altair import Undefined
from typing import List, Optional, Callable, Dict, Any
from inspect_ai.log import EvalLog, EvalLogInfo, read_eval_log
from inspect_ai.scorer import value_to_float
from dataclasses import dataclass, field


@dataclass
class VisualizationConfig:
    plot_fn: Callable[..., alt.Chart] = alt.Chart.mark_bar
    fig_title: Optional[str] = None
    plot_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    chart_properties: Dict[str, Any] = field(default_factory=dict)
    x_category: Optional[str] = None
    y_category: Optional[str] = None
    x_offset_category: Optional[str] = None
    y_offset_category: Optional[str] = None
    color_category: Optional[str] = None
    color_scheme: Optional[str] = None
    color_domain: Optional[List[str]] = None
    color_range: Optional[List[str]] = None
    color_legend: Optional[alt.Legend] = None
    opacity_category: Optional[str] = None
    opacity_legend: Optional[alt.Legend] = None
    layer_category: Optional[str] = None
    facet_category: Optional[str] = None
    facet_columns: Optional[int] = None
    h_concat_category: Optional[str] = None
    v_concat_category: Optional[str] = None
    shared_y_scale: bool = False
    tooltip_fields: Optional[List[alt.Tooltip]] = None
    titles: Optional[Dict[str, str]] = field(default_factory=dict)
    legend_config: Dict[str, Any] = field(default_factory=lambda: {
        "orient": "bottom", 
        "columns": 3, 
        "titleAlign": "center",
        "labelLimit": 1000
    })


class EvalVisualizer:
    """
    A class for visualizing evaluation logs using Altair charts.

    Attributes:
        eval_logs (List[EvalLog]): List of evaluation logs to visualize.
        value_to_float_fn (Callable): Function to convert score values to floats.
        df (pd.DataFrame): DataFrame containing processed evaluation data.

    Methods:
        __init__(eval_logs, value_to_float_fn, categorizers, rename_mappings, filter_sort_order): Initialize the visualizer.
        _create_dataframe(): Create a DataFrame from evaluation logs.
        _add_categories(source_columns, categorizer, apply_axis): Add categories to the DataFrame.
        visualize(...): Generate Altair charts based on the evaluation data.
    """

    def __init__(
        self,
        eval_log_infos: list[EvalLogInfo],
        value_to_float_fn: Callable[[Any], float] = value_to_float(),
        categorizers: Dict[str | tuple[str, ...], Callable] = None,
        rename_mappings: Dict[str, Dict[str, str]] = None,
        filter_sort_order: Dict[str, List[str]] = None,
    ):
        """
        Initialize the EvalVisualizer.

        Args:
            eval_logs (List[EvalLog]): List of evaluation logs.
            value_to_float_fn (Callable): Function to convert score values to floats.
            categorizers (Dict[str | tuple[str, ...], Callable]): Dictionary mapping column names 
                (or tuples of column names) to their categorizer functions. For multi-column 
                categorizers, use a tuple of column names as the key and set apply_axis=1.
            rename_mappings (Dict[str, Dict[str, str]]): Mappings for renaming categories.
                Keys are column names, values are dictionaries mapping old names to new names.
            filter_sort_order (Dict[str, List[str]]): Custom sort order for categories.
                Keys are column names, values are lists defining the desired order using the new names.
        """
        self.eval_log_infos = eval_log_infos
        self.eval_logs = self._get_eval_logs()
        self.value_to_float_fn = value_to_float_fn
        self.filter_sort_order = filter_sort_order or {}
        self.rename_mappings = rename_mappings or {}
        self.df = self._create_dataframe()

        # Add categories based on provided categorizers
        if categorizers:
            for columns, categorizer in categorizers.items():
                # If columns is a tuple, it's a multi-column categorizer
                apply_axis = 1 if isinstance(columns, tuple) else 0
                self._add_categories(columns, categorizer, apply_axis=apply_axis)

        self._process_dataframe()

    def _get_eval_logs(self) -> list[EvalLog]:
        return [read_eval_log(eval_log_info) for eval_log_info in self.eval_log_infos]

    def _get_log_dir(self, name: str) -> str:
        return "/".join(name.split("///")[1].split("/")[:-1])

    def _create_dataframe(self) -> pd.DataFrame:
        data = []
        for i, log in enumerate(self.eval_logs):
            if (
                log.status != "success" or not log.samples
            ):  # skip logs that are not successful or have no samples
                continue

            log_dir = self._get_log_dir(self.eval_log_infos[i].name)
            timestamp = self.eval_log_infos[i].mtime
            suffix = self.eval_log_infos[i].suffix
            run_id = log.eval.run_id
            task = log.eval.task
            task_args = log.eval.task_args
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
                                "task_args": task_args,
                                "dataset": dataset,
                                "model": model,
                                "scorer": scorer,
                                "value": self.value_to_float_fn(score.value),
                            }
                        )
        return pd.DataFrame(data)

    def _add_categories(
        self, 
        source_columns: str | tuple[str, ...] | list[str], 
        categorizer: Callable,
        apply_axis: int = 0
    ):
        # Convert tuple to list for pandas compatibility
        if isinstance(source_columns, tuple):
            source_columns = list(source_columns)
            
        if apply_axis == 1:
            # For multi-column categorizers, pass all columns as arguments
            categories = self.df[source_columns].apply(
                lambda x: categorizer(*x),
                axis=apply_axis
            )
        else:
            # For single-column categorizers, pass the value directly
            categories = self.df[source_columns].apply(categorizer)
            
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
                self.df[column] = pd.Categorical(
                    self.df[column], categories=order, ordered=True
                )
                # Filter the dataframe to keep only the values in the custom sort order
                self.df = self.df[self.df[column].isin(order)]

        # Sort the dataframe based on all columns with custom sort order
        sort_columns = [
            col for col in self.filter_sort_order.keys() if col in self.df.columns
        ]
        if sort_columns:
            self.df = self.df.sort_values(sort_columns)

    def visualize(self, config: VisualizationConfig):
        """
        Generate Altair charts based on the evaluation data.

        Args:
            config (VisualizationConfig): Configuration for the visualization
        
        Returns:
            alt.Chart: Altair (multi-)chart.
        """
        titles = config.titles or {}
        
        def get_title(category: str) -> str:
            # Strip any Altair type shorthand (e.g., ':O', ':N') from the category
            base_category = category.split(':')[0]
            return titles.get(base_category, base_category)
        
        charts = []
        df = self.df.copy()

        color_scale = alt.Scale()
        if config.color_scheme:
            color_scale = alt.Scale(scheme=config.color_scheme)
        if config.color_range:
            color_scale = alt.Scale(range=config.color_range)
        if config.color_domain:
            color_scale.domain = config.color_domain

        encoding = {
            "x": alt.X(config.x_category, title=get_title(config.x_category)),
            "y": alt.Y(config.y_category, stack=False, title=get_title(config.y_category)),
        }

        if config.tooltip_fields:
            encoding["tooltip"] = config.tooltip_fields

        if config.opacity_category:
            opacity_title = get_title(config.opacity_category)

            encoding["opacity"] = alt.Opacity(
                config.opacity_category,
                scale=alt.Scale(domain=[0, 1], range=[0.1, 1]),
                legend=config.opacity_legend if config.opacity_legend else alt.Legend(title=opacity_title),
            )

        base = alt.Chart(df).encode(**encoding)

        if config.color_category:
            color_title = get_title(config.color_category)

            base = base.encode(
                color=alt.Color(
                    config.color_category,
                    scale=color_scale,
                    legend=config.color_legend if config.color_legend else alt.Legend(
                        title=color_title,
                        labelLimit=1000  # You can also set it here specifically for color legend
                    )
                )
            )
        if config.x_offset_category:
            base = base.encode(xOffset=alt.XOffset(f"{config.x_offset_category}"))
        if config.y_offset_category:
            base = base.encode(yOffset=alt.YOffset(f"{config.y_offset_category}"))

        chart = config.plot_fn(base, **config.plot_fn_kwargs).properties(**config.chart_properties)

        if config.layer_category:
            chart = alt.layer(
                *[
                    chart.transform_filter(f"datum.{config.layer_category} == '{val}'")
                    for val in df[config.layer_category].unique()
                ]
            )

        if config.facet_category:
            facet_title = get_title(config.facet_category)
            chart = chart.facet(
                facet=alt.Facet(config.facet_category, title=facet_title),
                columns=config.facet_columns or alt.Undefined
            )

        charts.append(chart)

        if config.h_concat_category:
            charts = [
                alt.hconcat(
                    *[
                        c.transform_filter(
                            f"datum.{config.h_concat_category} == '{val}'"
                        ).properties(
                            title=f"{get_title(config.h_concat_category)}: {val}"
                        )
                        for val in df[config.h_concat_category].unique()
                    ],
                    resolve=alt.Resolve(scale={"y": "shared" if config.shared_y_scale else "independent"})
                )
                for c in charts
            ]

        if config.v_concat_category:
            charts = [
                alt.vconcat(
                    *[
                        c.transform_filter(
                            f"datum.{config.v_concat_category} == '{val}'"
                        ).properties(
                            title=f"{get_title(config.v_concat_category)}: {val}"
                        )
                        for val in df[config.v_concat_category].unique()
                    ],
                    resolve=alt.Resolve(scale={"y": "shared" if config.shared_y_scale else "independent"})
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
        if config.fig_title:
            combined_chart = combined_chart.properties(
                title=alt.TitleParams(text=config.fig_title, align="center", anchor="middle")
            )

        # Apply legend configuration if provided
        if config.legend_config:
            combined_chart = combined_chart.configure_legend(**config.legend_config)

        return combined_chart
