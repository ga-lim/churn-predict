# %%

import pandas as pd

from sklearn import model_selection
from sklearn import tree
from sklearn import linear_model
from sklearn import metrics

from feature_engine import discretisation, encoding

# %%

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

# %%
df = pd.read_csv('../data/abt_churn.csv')
df.head()


# %%
# Sample


oot = df[df['dtRef']==df['dtRef'].max()].copy()

df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train

# %%

features = df_train.columns[2:-1]

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
# Explore

df_analise = X_train.copy()
df_analise[target] = y_train.copy()

df_analise.head()
# %%

sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]

sumario.sort_values(by=['diff_rel'], ascending=False)

# %%

model = tree.DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# %%

feature_importance = (pd.Series(model.feature_importances_, index=X_train.columns)
                      .sort_values(ascending=False)
                      .reset_index())

feature_importance

# %%

feature_importance['acum.'] = feature_importance[0].cumsum()
feature_importance[feature_importance['acum.'] < 0.96]

# %%
best_features = (feature_importance[feature_importance['acum.'] < 0.96]['index']
                 .to_list())

best_features

# %%
# Modify


disc = discretisation.DecisionTreeDiscretiser(variables=best_features, random_state=42, cv=3, bin_output='bin_number', regression=False)
disc.fit(X_train[best_features], y_train)

x_train_transform = disc.transform(X_train[best_features])

onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)
onehot.fit(x_train_transform, y_train)

x_train_transform = onehot.transform(x_train_transform)
