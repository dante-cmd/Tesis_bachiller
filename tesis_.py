
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# activate plot theme
import qeds
qeds.themes.mpl_style();
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'

#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import shap

df = pd.read_csv('data_tesis.csv')
df['index'] = pd.to_datetime(df['index'])
df_1 = df.set_index('index').copy()
df_1.columns = ['IDE', 'PTF', 'TIPO CAMBIO', 'TIR', 'INVERSION PRIVADA' ,'INVERSION PUBLICA','APERTURA COMERCIAL']
df_2 = df_1.drop('TIR', axis=1)

var_x = 'TIPO CAMBIO'

df_to_corr = df_2[['IDE', var_x]].copy()
df_to_corr[var_x + '_SHIFT_1'] = df_to_corr[var_x].shift(1)
df_to_corr[var_x + '_SHIFT_2'] = df_to_corr[var_x].shift(2)
df_to_corr[var_x + '_SHIFT_3'] = df_to_corr[var_x].shift(3)

df_to_corr.dropna().corr()[['IDE']]

df_2[['IDE', 'TIPO CAMBIO']].plot(
    subplots = True, 
    figsize = (14, 16), 
    layout = (4, 2), color = 'k', alpha = 1, lw = 1, xlabel='Año') #.dropna().corr()['IDE'] #loc['1993':, :]

plt.show()
df_2.plot(
    subplots = True, 
    figsize = (14, 16), 
    layout = (4, 2), color = 'k', alpha = 1, lw = 1, xlabel='Año')

plt.show()

train_df = df_2.loc[:'2017', :]
test_df = df_2.loc['2018':, :]

features = ['PTF', 'TIPO CAMBIO','INVERSION PUBLICA','APERTURA COMERCIAL']
target = 'IDE'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

from sklearn.decomposition import PCA
from sklearn import metrics

pca = PCA()
qs =pca.fit_transform(df_1[features])
vexp = pca.explained_variance_ratio_
(vexp*100).round(1)

model_rf = RandomForestRegressor(random_state=123)

model_rf.fit(X_train, y_train)

model_rf.score( X_train, y_train)

(y_train - model_rf.predict(X_train) ).plot(kind = 'hist', title = 'Histograma de los Errores', figsize = (7, 6), density = True, alpha = 0.5)


metrics.mean_absolute_error(y_train, model_rf.predict(X_train) )


serie_importances = pd.Series(model_rf.feature_importances_*100, index = X_train.columns)
serie_importances.index = ['PTF', 'TIPO\nCAMBIO', 'INVERSION\nPUBLICA','APERTURA\nCOMERCIAL']
ax= serie_importances.sort_values().plot(
    kind = 'barh', 
    title = 'Feature Importances', 
    figsize = (10, 6), 
    color = 'k', 
    alpha = 0.5)
ax.set_xlabel('Porcentaje')
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))

lenght_train = model_rf.predict(X_train).shape[0]
lenght_test = model_rf.predict(X_test).shape[0]

ax.plot(df_2[target].values, color = 'k', lw = 1, marker = 'o', label = 'IDE REAL')
ax.plot(range(lenght_train), model_rf.predict(X_train),color = 'b', lw = 1, marker = 'o', label = 'IDE PREDICT INSAMPLE')
ax.plot(range(lenght_train, lenght_train + lenght_test),model_rf.predict(X_test),color = 'r', lw = 1, marker = 'o', label = 'IDE PREDICT OUTSAMPLE')
ax.legend()
plt.show()

mod_add_shap = shap.TreeExplainer(model_rf)
shap_values = mod_add_shap.shap_values(X_train)

shap.initjs()
shap.force_plot(mod_add_shap.expected_value, shap_values[0,:], X_train.iloc[0,:])

shap.initjs()
shap.force_plot(mod_add_shap.expected_value, shap_values, X_train)

shap.summary_plot(shap_values, X_train)

shap.initjs()
shap.dependence_plot("PTF", shap_values, X_train)

mod_add_shap = shap.Explainer(model_rf, X_train)
shap_values = mod_add_shap(X_train)
shap.initjs()
q = shap.plots.waterfall(shap_values[0, :])

shap_values[-1, :]

shap_values.values[:,2].round(2).mean()

mod_add_shap = shap.Explainer(model_rf, X_train)
shap_values = mod_add_shap(X_train)
shap.initjs()
shap.plots.scatter(shap_values[:, 'AP'])

mod_add_shap = shap.Explainer(model_rf, X_train)
shap_values = mod_add_shap(X_train)
shap.initjs()
shap.plots.scatter(shap_values[:, 'ctfp'])

# División entre 1993

train_df = df_2.loc[:'1993', :]
test_df = df_2.loc['1994':, :]

features = ['PTF', 'TIPO CAMBIO', 'INVERSION PUBLICA', 'APERTURA COMERCIAL'] #'irr',
target = 'IDE'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


ax = train_df[['IDE', 'APERTURA COMERCIAL']].plot(
    subplots = True, 
    figsize  =(10, 7),
    title = 'IDE e Inversión Pública',
    xlabel = 'Año', ylabel = 'Porcentaje')


model_rf_train = RandomForestRegressor(random_state=123)

model_rf_train.fit(X_train, y_train)

serie_importances_train = pd.Series(model_rf_train.feature_importances_*100, index = X_train.columns)


model_rf_test = RandomForestRegressor(random_state=123)

model_rf_test.fit(X_test, y_test)

serie_importances_test = pd.Series(model_rf_test.feature_importances_*100, index = X_test.columns)


df_importances = pd.concat([serie_importances_train, serie_importances_test], axis=1)
df_importances.columns = ['ANTES_1993', 'DESPUES_1993'] #.plot(kind = 'bar')
df_importances.index = ['PTF', 'TIPO\nCAMBIO', 'INVERSION\nPUBLICA','APERTURA\nCOMERCIAL', ] # 'TIR',
df_importances



df_importances.plot(kind = 'bar', figsize = (10, 7), rot = 0, title = 'Feature Importances')
plt.show()


from sklearn import (metrics, model_selection)

from sklearn.ensemble import RandomForestRegressor


df_1.head()


def model_rf_reg(df, vars_x, var_y, model, lag_x = 1):
    """
    col_effects must be in vars_x
    
    """
    df1 = df.reset_index(drop = True).copy()
    X = df1.loc[:, vars_x].shift(lag_x).dropna()
    y = df1.loc[lag_x:, var_y]
    
    model_rf = model(random_state=123)
    result = model_rf.fit(X, y)
    
    mse_tree_train = metrics.mean_squared_error(y, result.predict(X))

    print("mse tree train {}, num lag x {}".format(mse_tree_train, lag_x))

    y_predict = result.predict(X)
    error = y - y_predict

    return result.feature_importances_, y_predict, error


*_, error = model_rf_reg(
        df_1, 
        ['ctfp', 'xr',  'inversion_publica','AP'], #'irr',
        'IDE', 
        RandomForestRegressor, 
        lag_x=1
        )
fig, ax = plt.subplots()

ax.hist(error)
plt.show()


df_empty = pd.DataFrame()

for lag in range(4):
    feat_imp, *_ = model_rf_reg(
        df_1, 
        ['ctfp', 'xr',  'inversion_publica','AP'], #'irr',
        'IDE', 
        RandomForestRegressor, 
        lag_x=lag
        )
    ser_feat_imp = pd.Series(feat_imp, name=rf'Shift {lag}')
    df_empty = pd.concat([df_empty, ser_feat_imp], axis=1)
    
df_empty.index = ['PTF', 'TIPO\nCAMBIO', 'INVERSION\nPUBLICA','APERTURA\nCOMERCIAL'] # 'TIR',
df_empty_1 = df_empty.multiply(100)


ax_feat = df_empty_1.plot(
    kind = 'bar', 
    figsize = (10, 7), 
    colormap = 'jet', 
    rot =0, 
    title = 'Importancia de las Variables', ylabel = 'Porcentaje (%)'
    )



fig_feat = ax_feat.get_figure()


fig_feat.savefig('features_importances.png', dpi = 400, bbox_inches = 'tight')
