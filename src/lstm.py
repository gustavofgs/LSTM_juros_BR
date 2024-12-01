import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from skopt import BayesSearchCV
from scikeras.wrappers import KerasRegressor

# Função para criar o dataset com base no look_back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# Classe para dividir a série temporal em blocos para validação cruzada
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

# Função para criar o modelo LSTM
def lstm_model(X_train, neurons1=8, neurons2=1, n_steps=3, lr=0.001):
    model = Sequential()
    model.add(tf.keras.Input(shape=(n_steps, X_train.shape[2])))
    model.add(LSTM(neurons1, return_sequences=True))
    model.add(LSTM(neurons2))
    model.add(Dense(X_train.shape[2]))
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')
    return model

# Função para transformar os valores em prazos
def tenors(values):
    elements = np.zeros((values.shape[1], values.shape[0]))
    for i in range(values.shape[1]):
        elements[i] = values[:, i]
    return elements

# Função para calcular a Mean Directional Accuracy (MDA)
def mdaf(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))

# Função para obter as métricas de avaliação
def getmetrics(ytest, pred):
    mse = mean_squared_error(ytest, pred)
    mae = mean_absolute_error(ytest, pred)
    mape = mean_absolute_percentage_error(ytest, pred) * 100
    mda = mdaf(tf.constant(ytest), tf.constant(pred))
    metrics = [mse, mae, mape, mda]
    return metrics

# Função para otimizar os hiperparâmetros do modelo LSTM
def optimize_lstm_hyperparameters(look_back, X_train, y_train, lstm_model, param_grid, fit):
    model = KerasRegressor(model=lstm_model, neurons1=8, neurons2=2, n_steps=look_back)
    my_cv = BlockingTimeSeriesSplit(n_splits=3).split(y_train)
    model = BayesSearchCV(estimator=model, search_spaces=param_grid, fit_params=fit, cv=my_cv, n_jobs=1, verbose=0)
    model.fit(X_train, y_train)
    model.best_params_
    print('Best parameters found:', model.best_params_)
    return model

# Função para avaliar o modelo
def evaluate(number_of_iterations, scaler, look_back, X_train, y_train, X_test, y_test, val, forecast_step, best_neurons1, best_neurons2, best_batch):
    fitpredicted = []
    predicted = []
    consolidated_metrics = []
    i = 0
    y_train_copy = y_train.copy()
    y_test_copy = y_test.copy()
    while i < number_of_iterations:
        # Cria o modelo LSTM com os melhores hiperparâmetros
        model = lstm_model(X_train=X_train, neurons1=best_neurons1, neurons2=best_neurons2, n_steps=look_back)
        es2 = EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=1, verbose=0)
        mc2 = ModelCheckpoint('best_model2.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        
        # Treina o modelo
        model.fit(X_train, y_train_copy, initial_epoch=0, epochs=500, batch_size=best_batch, validation_split=val, callbacks=[es2, mc2], verbose=0)
        
        # Carrega o melhor modelo salvo
        best_model = load_model('best_model2.keras')
        
        # Faz previsões no conjunto de treinamento
        fitpred = best_model.predict(X_train, verbose=0)
        
        # Inverte a normalização dos dados
        y_train_inv = scaler.inverse_transform(y_train_copy.reshape(-1, y_train_copy.shape[-1]))
        fitpred_inv = scaler.inverse_transform(fitpred.reshape(-1, fitpred.shape[-1]))
        
        # Transforma os dados em tenores
        fitpred_tenor = tenors(fitpred_inv)
        y_train_tenor = tenors(y_train_inv)
        
        # Calcula as métricas de avaliação no conjunto de treinamento
        fitmetrics = []
        for j in range(len(fitpred_tenor)):
            fitmetrics.append(getmetrics(y_train_tenor[j], fitpred_tenor[j]))
        fitpredicted.append(fitpred_tenor)
        
        # Calcula a MDA no conjunto de treinamento
        mda_fit = np.mean([x[3] for x in fitmetrics])
        print("MDA: ", mda_fit)
        
        # Se a MDA for maior que 0.5, faz previsões no conjunto de teste
        if mda_fit > 0.5:
            prediction_steps = []
            for j in range(forecast_step):
                last_prediction = best_model.predict(X_test, verbose=0)
                last_prediction = np.reshape(last_prediction, (1, last_prediction.shape[-2], last_prediction.shape[-1]))
                X_test = np.vstack([X_test[1:], last_prediction[:, -X_test.shape[1]:, :]])
                print("Prediction ", j + 1, "Completed")
                prediction_steps.append(last_prediction)
            
            # Inverte a normalização dos dados de teste
            y_test_inv = scaler.inverse_transform(y_test_copy.reshape(-9, 9))
            pred_inv = scaler.inverse_transform(last_prediction.reshape(-9, 9))
            
            # Transforma os dados em tenores
            pred_tenor = tenors(pred_inv)
            y_test_tenor = tenors(y_test_inv)
            
            # Calcula as métricas de avaliação no conjunto de teste
            metrics = []
            for j in range(len(pred_tenor)):
                metrics.append(getmetrics(y_test_tenor[j], pred_tenor[j]))
            consolidated_metrics.append(metrics)
            predicted.append(pred_tenor)
            i += 1
            print("Iteration ", i, "Completed")
        else:
            print("Iteration ", i + 1, "Failed")
    return consolidated_metrics, y_train_tenor, y_test_tenor, fitpredicted, predicted