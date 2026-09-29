# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_csv('../data/abt_churn.csv')
df.head()


# %%

df.describe().T
