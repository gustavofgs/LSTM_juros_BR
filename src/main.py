from data import load_data, compile_prediction_results
from random_walk import forecast_RW, calculate_metrics
from dns import lamb, DNS_OLS, forecast_DNS_VAR, nelson_siegel_yields, calculate_metrics_dns
from lstm import create_dataset, lstm_model, optimize_lstm_hyperparameters, evaluate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Passo de previsão
forecast_step = 1
redefine_lambda = False
redefine_hyperparameters = True

# Carregar dados
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DI1_settle.xlsx')
dates, y, matu = load_data(file_path)

# Dividir dados em treino e teste
data = y
train_data = data[:int(len(data) * 0.7)]
test_data = data[int(len(data) * 0.7) + 1:]

# Random Walk
train_predictions = forecast_RW(train_data, forecast_step)
test_predictions = forecast_RW(test_data, forecast_step)
train_predictions = train_predictions[forecast_step:]
test_predictions = test_predictions[forecast_step:]
train_data = train_data[forecast_step:]
test_data = test_data[forecast_step:]
metrics_RW = calculate_metrics(test_data, test_predictions, forecast_step)

# DNS
if redefine_lambda:
    lambdas = lamb(train_data, matu)
    lambda_mean = np.mean(lambdas)
else:
    lambda_mean = 0.5
ts = DNS_OLS(y, matu, lambda_mean)
tsf = pd.DataFrame(ts)
predicted_factors = forecast_DNS_VAR(tsf, len(test_data))
predictions = nelson_siegel_yields(predicted_factors, matu[0], lambda_mean)
metrics_DNS = calculate_metrics_dns(test_data, predictions)

# LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(y)
train_data = scaled_data[:int(len(scaled_data) * 0.7)]
test_data = scaled_data[int(len(scaled_data) * 0.7) + 1:]
look_back = 10
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Otimizar hiperparâmetros do LSTM
if redefine_hyperparameters:
    neurons1 = [32,64,128,192,256]
    neurons2 = [32,64,128,192,256]
    batch = [16,32,64,128]
    param_grid = dict(neurons1=neurons1, neurons2=neurons2, batch_size=batch)
    fit = dict(epochs= [5,10,20,40,80])
    model = optimize_lstm_hyperparameters(look_back, X_train, y_train, lstm_model, param_grid, fit)
    best_neurons1 = model.best_params_['neurons1']
    best_neurons2 = model.best_params_['neurons2']
    best_batch = model.best_params_['batch_size']
else:
    best_neurons1 = 192
    best_neurons2 = 64
    best_batch = 16

# Avaliar modelo LSTM
number_of_iterations = 1
metrics, y_train_tenor, y_test_tenor, fitpredicted, predicted = evaluate(number_of_iterations, scaler, look_back, X_train, y_train, X_test, y_test, 0.2, forecast_step, best_neurons1, best_neurons2, best_batch)
metrics_avg = np.mean(metrics, axis=0)
metrics_avg_transposed = metrics_avg.T
metrics_LSTM = pd.DataFrame(metrics_avg_transposed, index=['MSE', 'MAE', 'MAPE', 'MDA'], columns=['1/12', '3/12', '6/12', '1', '1.5', '2', '3', '5', '10'])

# Compilar resultados das previsões
plot_data_df = compile_prediction_results(forecast_step, y_train_tenor, y_test_tenor, fitpredicted, predicted, train_predictions, test_predictions)

# Salvar métricas
metrics_RW.to_csv('metrics_RW.csv')
metrics_DNS.to_csv('metrics_DNS.csv')
metrics_LSTM.to_csv('metrics_LSTM.csv')
plot_data_df.to_csv('prediction_results.csv', index=False)
