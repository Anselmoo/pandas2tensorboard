"""Test of the Pandas2TensorBoard."""
import seaborn as sns

from pandas2tensorboard import pandas2tensorboard as p2t


def test_scatter():
    """Test of the scatter plot."""
    pt = p2t.Pandas2TensorBoard()
    pt.scatter_df(
        sns.load_dataset("anagrams"),
        x_axis="subidr",
        group="anagrams",
        remove_nan=True,
        remove_str=True,
    )
    pt.close()
    assert True


class TestRemove:
    """Test of removing columns and rows."""

    def test_remove_false(self):
        """Test of deactivated removing columns and rows."""
        pt = p2t.Pandas2TensorBoard()
        pt.regular_df(
            sns.load_dataset("gammas").drop(columns=["ROI"]),
            label="subidr",
            remove_nan=False,
            remove_str=False,
        )
        pt.close()
        assert True

    def test_remove_true(self):
        """Test of activated removing columns and rows."""
        pt = p2t.Pandas2TensorBoard()
        pt.regular_df(
            sns.load_dataset("planets"),
            label="planets",
            remove_nan=True,
            remove_str=True,
        )
        pt.close()
        assert True


class TestLabelGroup:
    """Test of the label and group."""

    def test_group(self):
        """Test of group and label."""
        pt = p2t.Pandas2TensorBoard()
        pt.regular_df(
            sns.load_dataset("anagrams"), group="group", label="label", remove_str=True
        )
        pt.close()
        assert True

    def test_only_label(self):
        """Test of only label."""
        pt = p2t.Pandas2TensorBoard()
        pt.regular_df(sns.load_dataset("anagrams"), label="label_only", remove_str=True)
        pt.close()
        assert True


class TestTimeSeries:
    """Test of time series."""

    def test_time_series_float(self):
        """Test of time series via timestamp."""
        pt = p2t.Pandas2TensorBoard()
        pt.timeseries_df(
            sns.load_dataset("attention"),
            time="score",
            label="attention",
            remove_nan=True,
            remove_str=True,
            time_convert=False,
        )
        pt.close()
        assert True

    def test_time_series_str(self):
        """Test of time series via float."""
        pt = p2t.Pandas2TensorBoard()
        pt.timeseries_df(
            sns.load_dataset("taxis"),
            time="pickup",
            label="taxis",
            remove_nan=True,
            remove_str=True,
        )
        pt.close()
        assert True
