# Yield Curve Forecasting

Este projeto prevê a curva de juros utilizando diferentes modelos: Random Walk, DNS e LSTM.

## Estrutura

- `data/`: Contém o conjunto de dados.
- `src/`: Contém o código-fonte dividido em módulos.
- `requirements.txt`: Lista as dependências.
- `README.md`: Breve documentação do projeto.

## Modelos

- **Random Walk**: Um modelo simples que assume que a melhor previsão para o futuro é o valor atual.
- **DNS (Dynamic Nelson-Siegel)**: Um modelo paramétrico que ajusta a curva de juros utilizando três fatores latentes.
- **LSTM (Long Short-Term Memory)**: Uma rede neural recorrente que captura dependências temporais de longo prazo nos dados.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.


