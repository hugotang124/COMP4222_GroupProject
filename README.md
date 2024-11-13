<h1> COMP 4222 Project </h1>
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
