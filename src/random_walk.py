import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Função para realizar a previsão Random Walk
def forecast_RW(data, forecast_step):
    repeated_values = np.zeros_like(data)
    for i in range(forecast_step, len(data)):
        # Atribui o valor do passo de previsão anterior à posição atual
        repeated_values[i] = data[i - forecast_step]

    print('Previsão Random Walk concluída')

    return repeated_values

# Função para calcular métricas de erro para a previsão
def calculate_metrics(test_data, test_predictions, forecast_step):
    mse_by_tenor = mean_squared_error(test_data, test_predictions, multioutput='raw_values')
    mae_by_tenor = mean_absolute_error(test_data, test_predictions, multioutput='raw_values')
    mape_by_tenor = mean_absolute_percentage_error(test_data[forecast_step:], test_predictions[forecast_step:], multioutput='raw_values') * 100
    metrics_RW = pd.DataFrame([mse_by_tenor, mae_by_tenor, mape_by_tenor], index=['MSE', 'MAE', 'MAPE (%)'], columns=['1/12', '3/12', '6/12', '1', '1.5', '2', '3', '5', '10'])

    print('Métricas calculadas')
    
    return metrics_RW