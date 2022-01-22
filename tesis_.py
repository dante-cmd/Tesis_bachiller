import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# activate plot theme
import qeds
qeds.themes.mpl_style();
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'

#from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import shap
import os
os.listdir()

##########################################################
#---------------- TESIS_DATA ---------------------------##
##########################################################

DF = pd.read_excel(r'data_tesis_1.xlsx', index_col='FECHA', parse_dates=True)
DF.IDE = DF.IDE/100

#DF_90 = DF.loc['1995':,].copy()
#DF_90.plot(subplots = True, layout = (5, 2), figsize = (12, 8)) #.drop('TIPO CAMBIO', axis = 1)
#DF_CORR = DF_90.drop(['INVERSION PRIVADA'], axis = 1).corr()
TAB_CORR = DF.corr()
names = DF.columns[~DF.columns.str.contains('IDE')]
for name in names:
    var_x = name
    df_to_corr = DF[['IDE', var_x]].copy()
    df_to_corr[var_x + '_SHIFT_1'] = df_to_corr[var_x].shift(1)
    df_to_corr[var_x + '_SHIFT_2'] = df_to_corr[var_x].shift(2)
    df_to_corr[var_x + '_SHIFT_3'] = df_to_corr[var_x].shift(3)
    print(df_to_corr.dropna().corr()[['IDE']])

DF.plot(subplots= True, layout = (4, 3))

DF.plot(
    subplots = True,
    figsize = (14, 16),
    layout = (4, 3), color = 'k', alpha = 1, lw = 1, xlabel='Año')

##########################################
## --------------- Model--------------###
#########################################
from sklearn import metrics

DF.columns
train_df = DF.loc[:'2017', :]
test_df = DF.loc['2018':, :]

features = ['POLI_STA', 'RegulatoryQuality', 'PTF', 'TIPO CAMBIO', 'PIB_MIN',
       'IPC', 'PBI_PERCAPITA', 'inversion_publica', 'AP', 'FEDFUNDS', 'VIX']
target = 'IDE'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

model_rf = RandomForestRegressor(random_state=123)

model_rf.fit(X_train, y_train)
print(model_rf.score( X_train, y_train))

(y_train - model_rf.predict(X_train) ).plot(kind = 'hist',
                                            title = 'Histograma de los Errores',
                                            figsize = (7, 6),
                                            density = True,
                                            alpha = 0.5)
print(metrics.mean_absolute_error(y_train, model_rf.predict(X_train) ))


serie_importances = pd.Series(model_rf.feature_importances_*100, index = X_train.columns)

ax= serie_importances.sort_values().plot(
    kind = 'barh',
    title = 'Feature Importances',
    figsize = (10, 6),
    color = 'k',
    alpha = 0.5)
ax.set_xlabel('Porcentaje')
fig = ax.get_figure()
fig.tight_layout()

fig, ax = plt.subplots(figsize = (12, 8))

lenght_train = model_rf.predict(X_train).shape[0]
lenght_test = model_rf.predict(X_test).shape[0]

ax.plot(DF[target].values, color = 'k', lw = 1, marker = 'o', label = 'IDE REAL')
ax.plot(range(lenght_train), model_rf.predict(X_train),color = 'b', lw = 1, marker = 'o', label = 'IDE PREDICT INSAMPLE')
ax.plot(range(lenght_train, lenght_train + lenght_test),model_rf.predict(X_test),color = 'r', lw = 1, marker = 'o', label = 'IDE PREDICT OUTSAMPLE')

ax.legend()
# SHAP VALUES
mod_add_shap = shap.TreeExplainer(model_rf)
shap_values = mod_add_shap.shap_values(X_train)

shap.initjs()
shap.force_plot(mod_add_shap.expected_value, shap_values[0,:], X_train.iloc[0,:])

shap.initjs()
shap.force_plot(mod_add_shap.expected_value, shap_values, X_train)

shap.summary_plot(shap_values, X_train)

shap.initjs()
shap.dependence_plot("PTF", shap_values, X_train, interaction_index = 'TIPO CAMBIO')

mod_add_shap = shap.Explainer(model_rf, X_train)
shap_values = mod_add_shap(X_train)
shap.initjs()
shap.plots.waterfall(shap_values[0, :])
