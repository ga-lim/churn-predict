# %%

import pandas as pd

from sklearn import model_selection
from sklearn import tree
from sklearn import pipeline
from sklearn import ensemble
from sklearn import metrics

from feature_engine import discretisation, encoding

import matplotlib.pyplot as plt

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


sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]

sumario.sort_values(by=['diff_rel'], ascending=False)

# %%

arvore = tree.DecisionTreeClassifier(random_state=42)

arvore.fit(X_train, y_train)

# %%

feature_importance = (pd.Series(arvore.feature_importances_, index=X_train.columns)
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

onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%
# Model

model = ensemble.RandomForestClassifier(random_state=42, min_samples_leaf=25, n_estimators=100)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', disc),
        ('OneHot', onehot),
        ('Model', model)
    ])

# %%
# Assets

# Treino

model_pipeline.fit(X_train[best_features], y_train)

y_train_predict = model_pipeline.predict(X_train[best_features])
y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]
roc_train = metrics.roc_curve(y_train, y_train_proba)

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print("Acurácia Treino: ", acc_train)
print("AUC Treino: ", auc_train)

# %%

# Teste

model_pipeline.fit(X_test[best_features], y_test)

y_test_predict = model_pipeline.predict(X_test[best_features])
y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]
roc_test = metrics.roc_curve(y_test, y_test_proba)

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print("Acurácia Teste: ", acc_test)
print("AUC Teste: ", auc_test)

# %%

# Out Of Time

model_pipeline.fit(oot[best_features], oot[target])

y_oot_predict = model_pipeline.predict(oot[best_features])
y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]
roc_oot = metrics.roc_curve(oot[target], y_oot_proba)

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

print("Acurácia Out Of Time: ", acc_oot)
print("AUC Out Of Time: ", auc_oot)

# %%

# Curva ROC

plt.Figure(dpi=400)
plt.Figure(figsize=(15,5))

plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0,1], [0,1], '--', color='black')

plt.grid(True)
plt.title('Curva ROC')
plt.legend([
    f'Treino: {auc_train*100:.2f}%',
    f'Teste: {auc_test*100:.2f}%',
    f'Out Of Time: {auc_oot*100:.2f}%'
])

plt.show()