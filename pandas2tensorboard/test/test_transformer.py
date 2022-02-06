import seaborn as sns

from pandas2tensorboard import pandas2tensorboard as p2t


def test_scatter():
    pt = p2t.Pandas2Tensorboard()
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
    def test_remove_false(self):
        pt = p2t.Pandas2Tensorboard()
        pt.regular_df(
            sns.load_dataset("gammas").drop(columns=["ROI"]),
            label="subidr",
            remove_nan=False,
            remove_str=False,
        )
        pt.close()
        assert True

    def test_remove_true(self):
        pt = p2t.Pandas2Tensorboard()
        pt.regular_df(
            sns.load_dataset("planets"),
            label="planets",
            remove_nan=True,
            remove_str=True,
        )
        pt.close()
        assert True


class TestLabelGroup:
    def test_group(self):
        pt = p2t.Pandas2Tensorboard()
        pt.regular_df(
            sns.load_dataset("anagrams"), group="group", label="label", remove_str=True
        )
        pt.close()
        assert True

    def test_only_label(self):
        pt = p2t.Pandas2Tensorboard()
        pt.regular_df(sns.load_dataset("anagrams"), label="label_only", remove_str=True)
        pt.close()
        assert True


class TestTimeSeries:
    def test_time_series_float(self):
        pt = p2t.Pandas2Tensorboard()
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
        pt = p2t.Pandas2Tensorboard()
        pt.timeseries_df(
            sns.load_dataset("taxis"),
            time="pickup",
            label="taxis",
            remove_nan=True,
            remove_str=True,
        )
        pt.close()
        assert True
