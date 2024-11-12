<h1> COMP 4222 Project </h1>
Welcome to our cryptocurrency project where we implemented a similar methodology to the research paper "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks". In this project, we used the following steps: </br>
1. Data Collection </br>
2. Graph Construction </br>
3. Feature Engineering </br>
4. Model Implementation and Modification </br>
5. Training and Evaluation </br>
</br>

Project Contributors: </br>
SHEKHAR, Shriyan </br>
KO, Sung Kit </br>
TANG, Siu Hang </br>
</br>

To get the dataset: </br>
```python fetch_data.py --symbols "PEPEUSDT,TRXUSDT,ADAUSDT,ATOMUSDT,BTCUSDT,VETUSDT,XLMUSDT,POLUSDT,AVAXUSDT,TONUSDT,SUIUSDT,RENDERUSDT,STXUSDT,LINKUSDT,OPUSDT,HBARUSDT,JUPUSDT,FTMUSDT,ALGOUSDT,FETUSDT,MKRUSDT,SHIBUSDT,UNIUSDT,LDOUSDT,THETAUSDT,SOLUSDT,IMXUSDT,TIAUSDT,ICPUSDT,WIFUSDT,APTUSDT,GRTUSDT,FLOKIUSDT,TAOUSDT,DOGEUSDT,WBETHUSDT,BCHUSDT,USDCUSDT,RUNEUSDT,WLDUSDT,BNBUSDT,OMUSDT,SEIUSDT,XRPUSDT,BONKUSDT,ETHUSDT,DAIUSDT,FILUSDT,DOTUSDT,ETCUSDT,INJUSDT,NEARUSDT,ENAUSDT,LTCUSDT,AAVEUSDT,ARBUSDT,PYTHUSDT" --start-date "2019-10-01" --end-date "2024-11-01" --interval "1h"```
</br>
</br>
To run the code for MTGNN (1st Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16"``` </br>
Alternatively (2nd Version - implemented by us): </br>
```python main.py --normalize "1" --time_interval "1h" --num_split "3"  --subgraph_size "2" --batch_size "16" --new```  </br>
</br>
The current Model uses the following: </br>
Learning Rate: 0.00001 </br>
Time Interval: 1h </br>
Sequence In Length: 168 (24 hours 7 days) </br>
Optimizer: Adam </br>
Horizon: 3 (default) </br>
