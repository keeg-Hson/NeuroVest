# 📈 CS2704-Market-Prediction-Algorithm

## 👨‍💻 Author

- Keegan Hutchinson [@keeg-Hson](https://www.github.com/keeg-Hson)

[➡️ Click here to download the final presentation (PowerPoint)](./CS2704_Live_Market_Prediction_Using_AI.pptx)

![CS2704: MrktPredAlgorithm](https://github.com/user-attachments/assets/d1b814eb-20b2-4832-b0b7-eeb985efeba8)

## Welcome!
The following project was both designed and implementtd to pose as a machine learning algorithm that seeks to predict both past, present, and future stock market behaviours (such as market spikes and crashes) wish the utilization of both built in technical indicators (such as RSI, moving average, and volatility calculations), as well as a Random Forest machine learning classifier model.

## Purpose:
This project was built based off of  live dataset valuations as extracted from the [Alpha Vantage API](https://www.alphavantage.co/documentation/), with the algorithmic function aiming to perform daily market predictions with accompanying confidence score valuations, log results, as well as associated trend valuations over time.

# Key Features:



### Real Time Data Ingestion:
From [Alpha Vantage API](https://www.alphavantage.co/documentation/).

### Technical Indicator Calculation:
Utilizing RSI, moving averages, volatiltiy, returns.

### Event Detection:
Labels major market movements (**crashes** and **spikes**).

### Random Forest ML Model:
implemented with utilization of a class balancing/confidence probabilities.

### Human Readable Predictions:
With built in alerts/warning symbols.

### Daily Scheduling:
Configured to run program at 6:00 PM on a daily basis for market set evaluation + dataset creation for training ML model.

### Confidence Trend Graph:
Visualize KPI displaying historic prediction probabilities.


### Prediction Logging: 
Intended for analysis + backtesting purposes.

 
-----
## Quick Start:
1. **Clone the repo**
   ```bash
   git clone https://github.com/keeg-Hson/CS2704-Market-Prediction-Algorithm
   cd CS2704-Market-Prediction-Algorithm

2. **Set up enviornment**
    **A) Install dependencies**
       
        pip install -r requirements.txt
    
    **B) Create .env file, add [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) here**
    
        ALPHA_VANTAGE_KEY=your_api_key_here

3. **Run program maually**
    
        python main.py

4. **Run with daily scheduler**
    
        python main.py  #Scheduler initializes automatically at 6:00 PM

-----

## Example Output:
### Graphs:
Saved to /graphs/
### Model: 
Stored in Market_Crash_Model.pkl
### Logs:
 Confidence prediction valuations stored in /prediction_log.txt
-----




