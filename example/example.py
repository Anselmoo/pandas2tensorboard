"""Example for the use of Pandas2TensorBoard."""

import seaborn as sns
import tensorboard

from pandas2tensorboard import pandas2tensorboard as p2t


pt = p2t.Pandas2TensorBoard()
pt.scatter_df(
    sns.load_dataset("anagrams"),
    x_axis="subidr",
    group="anagrams",
    remove_nan=True,
    remove_str=True,
)
pt.close()
