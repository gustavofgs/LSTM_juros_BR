import pandas as pd
import numpy as np

# Função para carregar os dados de um arquivo Excel
def load_data(file_path):
    y_full_df = pd.read_excel(file_path, names=['dates', 1/12, 3/12, 6/12, 1, 1.5, 2, 3, 5, 10])
    y_full = y_full_df.to_numpy()
    matu = np.array([[1/12, 3/12, 6/12, 1, 1.5, 2, 3, 5, 10]])
    dates = y_full[:, 0]
    y = y_full
    dates = np.array([y[:, 0]])
    y = np.delete(y, 0, 1)
    y = np.array(y, dtype=float)

    print('Data loaded')

    return dates, y, matu

# Função para compilar os resultados das previsões
def compile_prediction_results(forecast_step, y_train_tenor, y_test_tenor, fitpredicted, predicted, train_predictions, test_predictions):
    # Cria um dicionário para armazenar os dados
    data_dict = {
        'Time': [],
        'Tenor': [],
        'Type': [],
        'Value': []
    }

    # Define os tenores
    tenors = ['1/12', '3/12', '6/12', '1', '1.5', '2', '3', '5', '10']

    # Concatena os dados de treino e teste para cada tenor
    for i, tenor in enumerate(tenors):
        # Adiciona os dados de treino
        for j in range(len(y_train_tenor[i])):
            data_dict['Time'].append(j)
            data_dict['Tenor'].append(tenor)
            data_dict['Type'].append('Train Observed')
            data_dict['Value'].append(y_train_tenor[i][j])
        
        # Adiciona os dados de teste
        for j in range(len(y_test_tenor[i])):
            data_dict['Time'].append(len(y_train_tenor[i]) + forecast_step + j)
            data_dict['Tenor'].append(tenor)
            data_dict['Type'].append('Test Observed')
            data_dict['Value'].append(y_test_tenor[i][j])
        
        # Adiciona as previsões de treino do LSTM
        for j in range(len(fitpredicted[-1][i])):
            data_dict['Time'].append(j)
            data_dict['Tenor'].append(tenor)
            data_dict['Type'].append('Train Predicted (LSTM)')
            data_dict['Value'].append(fitpredicted[-1][i][j])
        
        # Adiciona as previsões de teste do LSTM
        for j in range(len(predicted[-1][i])):
            data_dict['Time'].append(len(fitpredicted[-1][i]) + forecast_step + j)
            data_dict['Tenor'].append(tenor)
            data_dict['Type'].append('Test Predicted (LSTM)')
            data_dict['Value'].append(predicted[-1][i][j])
        
        # Adiciona as previsões de teste do Random Walk
        for j in range(len(test_predictions[:, i])):
            data_dict['Time'].append(len(train_predictions[:, i]) + forecast_step + j)
            data_dict['Tenor'].append(tenor)
            data_dict['Type'].append('Test Predicted (RW)')
            data_dict['Value'].append(test_predictions[:, i][j])

    # Cria um DataFrame a partir do dicionário
    plot_data_df = pd.DataFrame(data_dict)
    return plot_data_df