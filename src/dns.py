# Este código foi adaptado de https://github.com/tomasrubin/yield-curve-forecasting

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Função que calcula a fórmula de Nelson-Siegel (DNS)
def DNS_formula(x, f, lambb):
    [l1, s1, c1] = f
    y = l1 + s1 * ((1 - np.exp(-lambb * x)) / (lambb * x)) + c1 * (((1 - np.exp(-lambb * x)) / (lambb * x)) - np.exp(-lambb * x))
    return y

# Função que calcula o erro quadrático da regressão OLS para DNS
def OLS_DNS_Error(data_i, lamb_i, tau_in):
    tau = tau_in.transpose()
    dummy = np.array(lamb_i * tau, dtype=float)
    # Calcula as colunas da matriz de regressão
    col2 = (np.ones(shape=(tau.size, 1)) - np.exp(-1 * dummy)) / dummy
    col3 = ((np.ones(shape=(tau.size, 1)) - np.exp(-1 * dummy)) / dummy) - np.exp(-1 * dummy)
    # Cria a matriz de regressão X
    X = np.hstack((np.ones((tau.size, 1)), col2, col3))
    # Ajusta o modelo OLS
    est = sm.OLS(data_i, X)
    est = est.fit()
    f = est.params
    squ_error = 0
    # Calcula o erro quadrático
    for i in range(tau.shape[0]):
        squ_error = squ_error + (DNS_formula(tau[i], f, lamb_i) - data_i[i]) ** 2
    return squ_error

# Função que encontra o melhor lambda para o modelo DNS
def lamb(y, tau):
    best_lamb = np.zeros(shape=(1, y.shape[0]))
    # Itera sobre cada série temporal
    for i in range(0, y.shape[0]):
        current_best = 1000
        y_present = y[i]
        # Testa diferentes valores de lambda
        for j in range(1, 2000):
            current_lamb = j / 1000
            fit = OLS_DNS_Error(y_present, current_lamb, tau)
            if fit < current_best:
                current_best = fit
                best_lamb[0, i] = current_lamb
    return best_lamb

# Função que realiza a regressão OLS para o modelo DNS
def DNS_OLS(data, tau_in, lamb_i):
    tau = tau_in.transpose()
    dummy = np.array(lamb_i * tau, dtype=float)
    f_concat = np.array(np.zeros(shape=(data.shape[0], 3)))
    # Itera sobre cada série temporal
    for i in range(0, data.shape[0]):
        y_i = np.array([data[i]]).transpose()
        # Calcula as colunas da matriz de regressão
        col2 = (np.ones(shape=(tau.size, 1)) - np.exp(-1 * dummy)) / dummy
        col3 = ((np.ones(shape=(tau.size, 1)) - np.exp(-1 * dummy)) / dummy) - np.exp(-1 * dummy)
        # Cria a matriz de regressão X
        X = np.hstack((np.ones((tau.size, 1)), col2, col3))
        # Ajusta o modelo OLS
        est = sm.OLS(y_i, X)
        est = est.fit()
        f = est.params
        f_concat[i] = f
    return f_concat

# Função que realiza a previsão usando o modelo VAR
def forecast_DNS_VAR(ts, pred):
    # Ajusta o modelo VAR
    model = VAR(ts)
    model_fitted = model.fit(1, method='mle')
    lag_order = model_fitted.k_ar
    # Realiza a previsão
    return model_fitted.forecast(ts.values[-lag_order:], pred)

# Função que calcula os yields usando os fatores de Nelson-Siegel
def nelson_siegel_yields(factors, tau, lamb):
    level, slope, curvature = factors[:, 0], factors[:, 1], factors[:, 2]
    yields = np.zeros((len(factors), len(tau)))
    # Calcula os yields para cada prazo
    for i in range(len(tau)):
        t = tau[i]
        yields[:, i] = level + slope * ((1 - np.exp(-lamb * t)) / (lamb * t)) + curvature * (((1 - np.exp(-lamb * t)) / (lamb * t)) - np.exp(-lamb * t))
    return yields

# Função que calcula a Acurácia Direcional Média (MDA)
def mdaf(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))

# Função que calcula as métricas de erro para o modelo DNS
def calculate_metrics_dns(test_data, predictions):
    mse = mean_squared_error(test_data, predictions, multioutput='raw_values')
    mae = mean_absolute_error(test_data, predictions, multioutput='raw_values')
    mape = mean_absolute_percentage_error(test_data, predictions, multioutput='raw_values') * 100
    # Calcula a MDA para cada prazo
    mda_by_tenor = [mdaf(test_data[:, i], predictions[:, i]) for i in range(test_data.shape[1])]
    # Cria um DataFrame com as métricas
    metrics_DNS = pd.DataFrame([mse, mae, mape, mda_by_tenor], index=['MSE', 'MAE', 'MAPE (%)', 'MDA'], columns=['1/12', '3/12', '6/12', '1', '1.5', '2', '3', '5', '10'])
    return metrics_DNS