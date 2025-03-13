import typing
from doespy.etl.steps.colcross.colcross import BaseColumnCrossPlotLoader, SubplotConfig
from doespy.etl.steps.colcross.base import is_match
from doespy.etl.steps.colcross.hooks import CcpHooks
from doespy.etl.steps.colcross.subplots.bar_hooks import BarHooks
from doespy.design.etl_design import MyETLBaseModel

from typing import Dict, List, Union

import os
from pdf2image import convert_from_path
import gossip
from matplotlib import pyplot as plt
import pandas as pd


class HLineConfig(MyETLBaseModel):

    y: float
    label: str
    value_col: str
    jp_query: typing.Optional[str] = None


class MyCustomSubplotConfig(SubplotConfig):

    # INFO: We extend the default config and add a new attribute
    grid: bool = True

    axhlines: List[HLineConfig] = []

    apply_bar_style_jp_query: str = None
    """Matches to enable bar-wide styling"""

    apply_line_jp_query: str = None


class MyCustomColumnCrossPlotLoader(BaseColumnCrossPlotLoader):

    # INFO: We provide a custom subplot config that extends the default config
    #       and override it here
    cum_subplot_config: List[MyCustomSubplotConfig]

    def setup_handlers(self):
        """:meta private:"""

        # NOTE: We can unregister function by name for a hook if needed
        # for x in gossip.get_hook(CcpHooks.SubplotPostChart).get_registrations():
        #    if x.func.__name__ == "ax_title":
        #        x.unregister()

        # NOTE: We can unregister all registered functions for a hook if needed
        # gossip.get_hook(SubplotHooks.SubplotPostChart).unregister_all()

        # install the class specific hooks
        MyCustomColumnCrossPlotLoader.blueprint().install()


    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        super().load(df, options, etl_info)

        # find all pdf files in the output directory and convert them to a png
        output_dir = self.get_output_dir(etl_info)

        pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
        pdf_paths = [os.path.join(output_dir, f) for f in pdf_files]

        for pdf_path in pdf_paths:
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            tup = convert_from_path(pdf_path)
            if len(tup) == 0:
                # print("Tupl", tup)
                print("Warning: No pages found in pdf", pdf_path)
                continue
            image = tup[0]
            image.save(os.path.join(output_dir, f"{filename}.png"))



@MyCustomColumnCrossPlotLoader.blueprint().register(CcpHooks.SubplotPostChart)
def apply_grid(
    ax: plt.Axes,
    df_subplot: pd.DataFrame,
    subplot_id: Dict[str, typing.Any],
    plot_config,
    subplot_config,
    loader,
):

   if subplot_config.grid:
      ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

@MyCustomColumnCrossPlotLoader.blueprint().register(CcpHooks.SubplotPostChart)
def apply_lines(
    ax: plt.Axes,
    df_subplot: pd.DataFrame,
    subplot_id: Dict[str, typing.Any],
    plot_config,
    subplot_config,
    loader,
):
    # ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45, rotation_mode='anchor')

    for axhline in subplot_config.axhlines:
        full_id = {'y': axhline.y, 'value_col': axhline.value_col }
        full_id.update({ **subplot_id })

        if axhline.jp_query is None or is_match(axhline.jp_query, full_id, 'axhline'):

            print(full_id, "full_id")
            line_config = subplot_config.artist_config(full_id, plot_config)
            print(line_config, "l")

            ax.axhline(y=axhline.y, linestyle='--', linewidth=1, **line_config)

            # put label on right side of the line
            ax.text(0.045, axhline.y, axhline.label, fontsize=8, ha='left', va='bottom',
                    transform=ax.get_yaxis_transform(), **line_config)
        else:
            print("No match for", axhline.jp_query, "in", full_id)

@MyCustomColumnCrossPlotLoader.blueprint().register(BarHooks.SubplotPost)
def apply_bar_styling(ax, df_subplot, subplot_id, metric, plot_config, subplot_config, chart, group_ticks, bar_ticks):

    if subplot_config.apply_bar_style_jp_query is None:
        return

    bars = {}
    for (
        part_value,
        part_error,
        position,
        bar_group_id,
        bar_id,
        bar_part_id,
    ) in chart.for_each(df_subplot, metric=metric, subplot_id=subplot_id):
        bar_and_group = bar_id.copy()
        bar_and_group.update(bar_group_id)
        frozen_bar = frozenset(bar_and_group.items())
        if frozen_bar not in bars.keys():
            bars[frozen_bar] = {
                "position": position,
                "value": part_value
            }
        else:
            bars[frozen_bar]["value"] += part_value

    for frozen_bar, bar in bars.items():
        full_id = {**subplot_id, **dict(frozen_bar)}
        full_id["bar_style"] = True

        if is_match(subplot_config.apply_bar_style_jp_query, full_id, 'bar_style'):
            bar_config = subplot_config.artist_config(full_id, plot_config)

            ax.bar(
                bar["position"].bar_center_pos,
                bar["value"],
                width=chart.bar_width,
                **bar_config
            )

@MyCustomColumnCrossPlotLoader.blueprint().register(BarHooks.SubplotPost)
def draw_line_on_bar_groups(ax, df_subplot, subplot_id, metric, plot_config, subplot_config, chart, group_ticks, bar_ticks):

    if subplot_config.apply_line_jp_query is None:
        return

    points_eps_map = {
        "eps3": 1.7,
        "eps5": 1.83,
        "eps7": 1.9,
        "eps10": 2.0,
        "eps15": 2.25,
        "eps20": 2.5,
    }

    bar_groups = {}
    for (
        part_value,
        part_error,
        position,
        bar_group_id,
        bar_id,
        bar_part_id,
    ) in chart.for_each(df_subplot, metric=metric, subplot_id=subplot_id):
        frozen_bar_group = frozenset(bar_group_id.items())

        # TODO: Maybe some jp_except
        if "budget_name" not in bar_id or bar_id["budget_name"] not in points_eps_map:
            continue

        if frozen_bar_group not in bar_groups.keys():
            bar_groups[frozen_bar_group] = [{
                "position": position,
                "bar_id": bar_id
            }]
        else:
            if bar_id not in [bar["bar_id"] for bar in bar_groups[frozen_bar_group]]:
                bar_groups[frozen_bar_group].append({
                    "position": position,
                    "bar_id": bar_id
                })

    for frozen_bar_group, bars in bar_groups.items():
        full_id = {**subplot_id, **dict(frozen_bar_group)}

        if is_match(subplot_config.apply_line_jp_query, full_id, 'line_overlay'):

            # get x,y positions
            x_pos = [bar["position"].bar_center_pos for bar in bars]
            y_pos = [points_eps_map[bar["bar_id"]["budget_name"]] for bar in bars]

            line_width = 1.2
            ax.hlines(y=y_pos, xmin=[x - line_width for x in x_pos], xmax=[x + line_width for x in x_pos], color='black', linestyle=(0, (2, 1)))


# This could probably be done in a more elegant way
@MyCustomColumnCrossPlotLoader.blueprint().register(BarHooks.SubplotPost)
def map_cohere_bottom(ax, df_subplot, subplot_id, metric, plot_config, subplot_config, chart, group_ticks, bar_ticks):

    if subplot_config.apply_line_jp_query is None:
        return

    points_eps_map = {
        "eps3": 1.7,
        "eps5": 1.83,
        "eps7": 1.9,
        "eps10": 2.0,
        "eps15": 2.25,
        "eps20": 2.5,
    }
    system_name = "cohere"

    bar_groups = {}
    for (
        part_value,
        part_error,
        position,
        bar_group_id,
        bar_id,
        bar_part_id,
    ) in chart.for_each(df_subplot, metric=metric, subplot_id=subplot_id):
        frozen_bar_group = frozenset(bar_group_id.items())

        # TODO: Maybe some jp_except
        if "budget_name" not in bar_id or bar_id["budget_name"] not in points_eps_map:
            continue
        if "system_name" not in bar_id or bar_id["system_name"] != system_name:
            continue

        if frozen_bar_group not in bar_groups.keys():
            bar_groups[frozen_bar_group] = [{
                "position": position,
                "bar_id": bar_id
            }]
        else:
            if bar_id not in [bar["bar_id"] for bar in bar_groups[frozen_bar_group]]:
                bar_groups[frozen_bar_group].append({
                    "position": position,
                    "bar_id": bar_id
                })

    for frozen_bar_group, bars in bar_groups.items():
        full_id = {**subplot_id, **dict(frozen_bar_group)}

        if is_match(subplot_config.apply_line_jp_query, full_id, 'line_overlay'):

            # get x,y positions
            x_pos = [bar["position"].bar_center_pos for bar in bars]
            y_pos = [points_eps_map[bar["bar_id"]["budget_name"]] for bar in bars]

            line_width = 1.2
            ax.hlines(y=y_pos, xmin=[x - line_width for x in x_pos], xmax=[x + line_width for x in x_pos], color='black', linestyle=(0, (2, 1)))

# @MyCustomColumnCrossPlotLoader.blueprint().register(CcpHooks.SubplotPostChart)
# def fix_ylims(ax, df_subplot, subplot_id, plot_config, subplot_config, loader):
#     if subplot_id['subplot_row_idx'] == 1:
#         ax.set_ylim([0, 10000000])
#     if subplot_id['subplot_row_idx'] == 0 and (subplot_id['subplot_col_idx'] == 0 or subplot_id['subplot_col_idx'] == 1):
#         ax.set_ylim([0, 20])

@MyCustomColumnCrossPlotLoader.blueprint().register(CcpHooks.FigPost)
def fig_legend(fig, axs, df_plot, plot_id, plot_config, loader):
    fig.subplots_adjust(wspace=0.15, hspace=0.25)