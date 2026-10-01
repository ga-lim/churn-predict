# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection


# %%
df = pd.read_csv('../data/abt_churn.csv')
df.head()


# %%
# Sample


oot = df[df['dtRef']==df['dtRef'].max()].copy()

df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train

# %%

features = df_train[2:-1].columns

target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    random_state=42, 
                                                                    test_size=0.2,
                                                                    stratify=y)
# %%

print('Taxa da variável resposta Treino', y_train.mean())
print('Taxa da variável resposta Teste', y_test.mean())
# %%
