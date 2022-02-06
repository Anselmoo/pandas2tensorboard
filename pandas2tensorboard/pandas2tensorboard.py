"""Transforms pandas dataframes into TensorBoard compatible summaries."""
try:
    import modin.pandas as pd

except ImportError:
    import pandas as pd

from pathlib import Path
from typing import MutableMapping
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter


class Pandas2TensorBoard:
    """Pandas2TensorBoard."""

    def __init__(
        self,
        log_dir: MutableMapping[Path, str] = None,
        comment: str = "",
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ) -> None:
        r"""Initialize the Pandas2TensorBoard.

        For more info about the arguments, see:
        https://pytorch.org/docs/stable/tensorboard.html

        Args:
            log_dir (MutableMapping[Path, str], optional): Name of the `log-directory`.
                 Defaults to None.
            comment (str, optional): Comment for the `log-directory`. Defaults to "".
            purge_step (int, optional): Purge step in case of crashing. Defaults to
                 None.
            max_queue (int, optional): Maximum size of the queue. Defaults to 10.
            flush_secs (int, optional): Flush after seconds. Defaults to 120.
            filename_suffix (str, optional): Suffix added to all event filenames.
                 Defaults to "".
        """
        self.log_dir = log_dir
        self.comment = comment
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix
        self.initialize()

    def initialize(self) -> None:
        """Initialize the tensorboard writer."""
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            comment=self.comment,
            purge_step=self.purge_step,
            max_queue=self.max_queue,
            flush_secs=self.flush_secs,
            filename_suffix=self.filename_suffix,
        )

    def regular_df(
        self,
        df: pd.DataFrame,
        group: str = None,
        label: str = "",
        remove_nan: bool = False,
        remove_str: bool = False,
    ) -> None:
        """Convert a pandas dataframe to a tensorboard timeseries.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            group (str, optional): Group name for the labels. Defaults to None.
            label (str, optional): Name of the label. Defaults to "".
            remove_nan (bool, optional): Remove rows with `NaNs` . Defaults to False.
            remove_str (bool, optional): Remove columns with `str`. Defaults to False.
        """
        df = self.df_cleaning(df=df, remove_nan=remove_nan, remove_str=remove_str)
        tag = self.create_tag(group=group, label=label)

        for i, val in enumerate(df.to_dict(orient="records")):
            self.writer.add_scalars(main_tag=tag, tag_scalar_dict=val, global_step=i)

    def timeseries_df(
        self,
        df: pd.DataFrame,
        time: str,
        time_convert: bool = True,
        group: str = None,
        label: str = "",
        remove_nan: bool = False,
        remove_str: bool = False,
    ) -> None:
        """Convert a pandas dataframe to a tensorboard timeseries.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            time (str): Column name of the time column.
            time_convert (bool, optional): Convert str `timestamp` to `float`.
                 Defaults to True.
            group (str, optional): Group name for the labels. Defaults to None.
            label (str, optional): Name of the label. Defaults to "".
            remove_nan (bool, optional): Remove rows with `NaNs` . Defaults to False.
            remove_str (bool, optional): Remove columns with `str`. Defaults to False.
        """
        if time_convert:
            df[time] = pd.to_datetime(df[time]).apply(lambda x: x.timestamp())
        df = self.df_cleaning(df=df, remove_nan=remove_nan, remove_str=remove_str)
        tag = self.create_tag(group=group, label=label)
        df_x, df_y = self.split_df(df=df, col=time)
        for i, val_time in enumerate(zip(df_x.values, df_y.to_dict(orient="records"))):
            self.writer.add_scalars(
                main_tag=tag,
                tag_scalar_dict=val_time[1],
                global_step=i,
                walltime=val_time[0],
            )

    def scatter_df(
        self,
        df: pd.DataFrame,
        x_axis: str,
        group: str = None,
        remove_nan: bool = False,
        remove_str: bool = False,
    ) -> None:
        """Convert a pandas dataframe to a tensorboard scatter plot.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            x_axis (str): Name of the `x-axis` column.
            group (str, optional): Group name for the labels. Defaults to None.
            remove_nan (bool, optional): Remove rows with `NaNs` . Defaults to False.
            remove_str (bool, optional): Remove columns with `str`. Defaults to False.
        """
        df = self.df_cleaning(df=df, remove_nan=remove_nan, remove_str=remove_str)
        tag = self.create_tag(group=group, label=x_axis)
        df_x, df_y = self.split_df(df=df, col=x_axis)

        for x_dict, y_dict in zip(df_x.to_dict(), df_y.to_dict(orient="records")):
            self.writer.add_hparams({tag: x_dict}, y_dict)

    def close(self) -> None:
        """Close the tensorboard writer."""
        self.writer.close()

    @staticmethod
    def split_df(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a dataframe into two dataframes.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            col (str): Name of the column to split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple of the two splitted dataframes.
        """
        return df[col], df.drop(columns=[col])

    @staticmethod
    def df_cleaning(
        df: pd.DataFrame, remove_nan: bool, remove_str: bool
    ) -> pd.DataFrame:
        """Remove nan and objects from DataFrame.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            remove_nan (bool, optional): Remove rows with `NaNs` . Defaults to False.
            remove_str (bool, optional): Remove columns with `str`. Defaults to False.

        Returns:
            pd.DataFrame: [description]
        """
        _df = df.copy()
        if remove_nan:
            _df = _df.dropna(axis=0, how="any")
        if remove_str:
            _df = _df.select_dtypes(include=["number", "datetime64[ns]"])
        return _df

    @staticmethod
    def create_tag(label: str, group: str = None) -> str:
        """Create tag for tensorboard.

        Args:
            label (str): Label name
            group (str, optional): Group name. Defaults to None.

        Returns:
            str: Tag name as concatenation of group (optional) and label
        """
        return f"{group}/{label}" if group else label
