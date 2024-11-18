<h1> Cryptocurrency Price Forecasting </h1>
In the rapidly evolving landscape of cryptocurrency markets, effective price forecasting remains a significant challenge due to the inherent volatility and complex interdependencies among various cryptocurrencies. This project aims to adapt a state-of-the-art framework, "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks," to the specific context of cryptocurrency datasets. Our objective is to leverage Graph Neural Networks (GNNs) to model the underlying spatial and temporal dependencies inherent in these datasets, potentially revealing complex relationships and trends that traditional methods may overlook. Our dataset consists of hourly and daily data. In this project, we used the following steps: </br>

1. Data Collection </br>
2. Graph Construction </br>
3. Feature Engineering </br>
4. Model Implementation and Modification </br>
5. Training and Evaluation </br>

We implement a Multivariate Time Series Graph Neural Network (MTGNN) structure tailored for cryptocurrency forecasting. Our MTGNN framework is designed to effectively capture the intricate spatial and temporal dependencies inherent in cryptocurrency price movements. To enhance our model's predictive capability, we create additional features linked to various aspects of the cryptocurrency market. After processing the data, the MTGNN has layers such as Graph Learning, Mix Hop Propagation, Dilated Inception, Attention Mechanism and more. </br>

To compare this framework, we have also coded several baseline models such as GAT, GCN, TGCN, A3TGCN and DCGNN.
</br>

Project Contributors: </br>
SHEKHAR, Shriyan </br>
KO, Sung Kit </br>
TANG, Siu Hang </br>
</br>

Files: </br>
```data_loader.py``` – involves standardizing the data, creating batches and adding features to the cryptocurrency raw data set. </br>
```fetch_data.py``` – This script is designed to retrieve data for each cryptocurrency through the BINANCE API. </br>
```old_layer.py``` - This module defines custom neural network layers used in the model architecture. This is the first version of the layers of MTGNN implemented by us. </br>
```layer.py``` - This module defines custom neural network layers used in the model architecture. This is the second version of the layers of MTGNN implemented by us. </br>
```old_net.py``` - This module implements the GTNet (Graph Temporal Network) architecture for multivariate time series forecasting using graph neural networks. This will be paired with old_layer.py. </br>
```net.py``` - This module implements the MTGNN (Multivariate Time Series Forecasting Graph Neural Network) architecture for predicting multivariate time series data using graph neural networks. This will be paired with layer.py. </br>
```main.py``` - This script serves as the entry point for the application. It orchestrates the training and evaluation process of the model. It also contains a parser to assign command-line arguments for configuring model parameters. </br>
```optimizer.py``` – This module defines a customizable optimizer class that supports various optimization methods, gradient clipping, and dynamic learning rate adjustment based on validation performance metrics. </br>
```requirements.txt``` – This file contains the requirements needed to run the program. </br>
```baseline-models.ipynb``` – Contains all the implemented baseline models and its results. </br>
```test_processing.py``` - Python script to get the MAPE for each cryptocurrency after getting results from MTGNN </br>


To get the dataset: </br>
```python fetch_data.py --symbols "PEPEUSDT,TRXUSDT,ADAUSDT,ATOMUSDT,BTCUSDT,VETUSDT,XLMUSDT,POLUSDT,AVAXUSDT,TONUSDT,SUIUSDT,RENDERUSDT,STXUSDT,LINKUSDT,OPUSDT,HBARUSDT,JUPUSDT,FTMUSDT,ALGOUSDT,FETUSDT,MKRUSDT,SHIBUSDT,UNIUSDT,LDOUSDT,THETAUSDT,SOLUSDT,IMXUSDT,TIAUSDT,ICPUSDT,WIFUSDT,APTUSDT,GRTUSDT,FLOKIUSDT,TAOUSDT,DOGEUSDT,WBETHUSDT,BCHUSDT,RUNEUSDT,WLDUSDT,BNBUSDT,OMUSDT,SEIUSDT,XRPUSDT,BONKUSDT,ETHUSDT,DAIUSDT,FILUSDT,DOTUSDT,ETCUSDT,INJUSDT,NEARUSDT,ENAUSDT,LTCUSDT,AAVEUSDT,ARBUSDT,PYTHUSDT" --start-date "2019-10-01" --end-date "2024-11-01" --interval "1h"```
</br>
</br>
To run the code for MTGNN - Multiple Features (1st Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16"``` </br>
Alternatively - Multiple Features (2nd Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16" --new```  </br>
To run the code for MTGNN - One Feature (Close Price) (1st Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16" --one_feature```  </br>
Alternatively - One Feature (2nd Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16" --new --one_feature```  </br>
To add an attention layer, simply add this at the end of the line: </br>
```--attention_layer```
</br>

The Current Model uses the following: </br>
Learning Rate: 0.001 </br>
Epochs: 16 </br>
Time Interval: 1h, 1d </br>
Sequence In Length: 168 (24 hours 7 days) for 1h, and 7 for 1d </br>
Optimizer: AdamW </br>
Horizon: 3 (default) </br>
