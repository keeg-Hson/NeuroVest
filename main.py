#-----------MARKET PREDICTION ALGORITHM-----------
#ADDITIONAL: INSTALL LIBRARIES ON LOCAL MACHINE (OR FROM TERMINAL TO REPO DIRECTLY? LOOK INTO THIS)

#___________USE OF ALPHA VANTAGE API TO PULL LIVE STOCK VALUATION FIGURES, TRAIN MODEL OFF OF THESE VALUATIONS (RANDOM FOREST MODEL)________
#ADDITIONALLY: BUILD IN FUCNTIONALITY THAT CPMPARES PREDICTED VALUATIONS WITH REAL TIME ONES/FUTURE ONES. THIS COULD BE ACHEVED WITH A GRAPHICAL VISUALIZATION OF PREDICTED VS. REAL TIME VALUATIONS

#BOOTING PROGRAM PROMPT
print("Welcome! Booting program now, please wait momemntarily...")

#LIBRARIES REQD'
import requests #PULLS STOCK MARKET DATA
import pandas as pd #ENHANCED DATA MANIPULATION LAYER
import time #PAUSE/TIMIING PROTOCOL: NECESSARY FOR LIVE VALUATIONS
time.sleep(12)  # wait 12 seconds between requests


#ADDITIONAL 
import numpy as np #ENHANCED NUMERICAL HANDLING
import matplotlib.pyplot as plt #GRAPH STATS
import sklearn.ensemble #ML MODEL TRAINING EVALUATING ACCURACY
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier #ML MODEL (ADDITIONAL)
import seaborn as sns #DATA VISUALIZATION
import joblib #SAVE/LOAD MODEL, GIVE USER CAPABILITY TO RUN ACROSS VARIOUS SESSIONS USING PRESET METRICS
import os #FILE MANAGEMENT
from dotenv import load_dotenv #DEALS WITH API KEY
import schedule #DAILY SCHEDULER
print("CWD:", os.getcwd())

from sklearn.utils import resample #DEALS WITH IMBALANCED DATASETS
import matplotlib.pyplot as plt #GRAPHICAL VISUALIZATION


#Alpha Vantage API key configuration
load_dotenv(dotenv_path="/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/.env") #"/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/AV-API-key.env" 
api_key = os.getenv("ALPHA_VANTAGE_KEY")
print(f"DEBUG: Loaded API key: {api_key}")

#DEBUG: check if API key is loaded
if not api_key:
    #raise ValueError("ERROR: API key not found. is it in your .env file?")
    print("DEBUG: API key not found! check .env file or file path")
else:
    print ("DEBUG: API Key loaded successfully!")

#Features
def get_feature_list():
    return [
        "MA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Width", "OBV", "Vol_Ratio", "Price_Momentum_10", "Acceleration",
        "RSI", "RSI_Delta", "ZMomentum",
        "Return_Lag1", "Return_Lag3", "Return_Lag5",
        "RSI_Lag_1", "RSI_Lag_3", "RSI_Lag_5"
    ]


#logging function to log predictions to a CSV file
import csv
from datetime import datetime
def log_prediction_to_file(prediction, class_probs, file_path="daily_predictions.csv"):
    headers = ["timestamp", "prediction", "normal_prob", "crash_prob", "spike_prob"]

    readable_label = "NORMAL" if prediction == 0 else "CRASH" if prediction == 1 else "SPIKE"


    data = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prediction,
        readable_label,
        round(class_probs[0] * 100, 2),
        round(class_probs[1] * 100, 2),
        round(class_probs[2] * 100, 2),
    ]

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

#-----------GENERAL PSEUDOCODE/HIERARCHICAL LAYOUT-----------


#CRITERIA/FUNCTIONAL COMPONENTS

#1. DATA INGESTION
#-DOWNLOAD DAILY/LIVE STOCK VALUATION FIGURES. TO BE ACCOMLISHED VIA. USE OF DAILY OHLCV FROM ALPHA VANTAGE
#--SCHEDULE DAILY JOB (VIA SCHEDULE)
#--FETCHING OF LATEST SPY DATA (VIA ALPHA VANTAGE API): *COMPLETED*
#---THIS IS TO FETCH CURRENT MARKET VALUATION VARIABLES, DAILY ADJUSTED OHLCV VALUATIONS 
#----VIA USE OF ***TIME_SERIES_DAILY_ADJUSTED*** ENDPOINT, RETURNING OHLCV VALUATIONS FROM AV API
def fetch_ohlcv(symbol="SPY", interval='1min', outputsize='full', api_key=None): #for testing: outputsize: (thing),  #add interval='1min' if it fails
    
    #Fetch daily OHLCV data from Alpha Vantage API
    print('Fetching OHLCV data valuations...')
    url=f"https://www.alphavantage.co/query" #THIS LINK MIGHT BE BROKEN

    params = {
        "function": "TIME_SERIES_DAILY", #ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
        #"function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "apikey": api_key,
        "outputsize": outputsize, #much smaller data set, may be good for avoiding overwhelming AI
        "datatype": "json"
    }

#    params = {
#        "function": "TIME_SERIES_INTRADAY", #ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
#        "symbol": "SPY",
#        "interval": "1min",
#        "apikey": api_key,
#        "outputsize": "compact",
#        "datatype": "json"
#    }
    print(f"DEBUG: API request params: {params}")
    print(f"DEBUG: API request URL: {url}?{requests.compat.urlencode(params)}")

    # Make the API request
    response = requests.get(url, params=params)
    print(f"DEBUG: API response status code: {response.status_code}")
    #-------
    #response=requests.get(url, params=params)

    #parse .json response
    data=response.json()
    print(f"DEBUG: API response: {data}")


    #DEBUG: check if API response was successful
    if response.status_code != 200:
        print(f"ERROR: API request failed with status code {response.status_code}")
        return None
    
    #check rate limit/invalid response time
    if "Note" in data:
        print("ERROR: API rate limit exceeded. Try again later.")
        return None
    #CHECK FOR INVALID/UNEXPECTED RESPONSE FORMAT
    time_series_key=[k for k in data.keys() if 'Time Series' in k]
    if not time_series_key:
        print("ERROR: Time Series data not found in API response")
        return None
    
    key = time_series_key[0]
    raw_df = pd.DataFrame.from_dict(data[key], orient='index')
    raw_df=raw_df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",        
        "5. volume": "Volume",
    })
    # Convert "sting" valuations to "floats"
    raw_df = raw_df.astype(float)

    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    print('DATA PARSED/EXTRACTED SUCCESSFULLY!')
    return raw_df

    # DEBUG: print first couple rows of DataFrame
    print(f"DEBUG: Parsed DataFrame head:\n{raw_df.head()}")

    raw_df.index =pd.to_datetime(raw_df.index)
    raw_df=raw_df.sort_index()
    print('DATA PARSED/EXTRACTED SUCCESSFULLY!')
    return raw_df

#new: test call
if __name__ == "__main__":
    print("DEBUG: Starting program...")
    df = fetch_ohlcv(symbol="SPY", api_key=api_key)
    if df is not None:
        print("DEBUG: Data fetched successfully!")
        print(df.head())
    else:
        print("ERROR: Failed to fetch data.")

#2. FEATURE ENGINEERING
#-STUCTURES ACTUAL SET FUNCTIONALITY OF ALGORITHMS TECHNICAL FEATURES
#--RSI, MACID, MOVING AVERAGES, VOLATILITY, RETURNS (OR OTHER RELEVANT INDICATORS)
#---ONLY MOST RECENT ROW NEEDS COMPUTATION: REDUCES REDUNCANCY + MITIGATES POTENTIAL MODEL OVERFITTING
#---***RECALCULATION OF FEATURE COLUMNS + FEATURE CONSISTENCY *CRITICAL* FOR VALID MODEL OUTPUT, AS IS TO BE SOLE BASIS OF OUR BELOW RENDITIONS!****
def calculate_technical_indicators(df):
    #this function intends to deal with adding our extracted moving averages, returns, RSI, etc. etc. etc.: effectively all the above metrics
    df["Return"] = df['Close'].pct_change()
    df["MA_20"] = df['Close'].rolling(window=20).mean()
    df["Volatility"] = df['Return'].rolling(window=20).std()

    #moving avg

    #Exponential Moving Average (EMA) 
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean() #Short term EMA (12)
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean() #Medium term EMA (26)

    #MACD, signal, histogram (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26'] #MACD line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean() #Signal line
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal'] #MACD histogram

    #Bollinger Bands
    df["BB_Mid"] = df['Close'].rolling(window=20).mean() #Middle band (20-day SMA)
    df["BB_Upper"] = df["BB_Mid"] + (df['Close'].rolling(window=20).std() * 2) #Upper band (20-day SMA + 2*std dev)
    df["BB_Lower"] = df["BB_Mid"] - (df['Close'].rolling(window=20).std() * 2) #Lower band (20-day SMA - 2*std dev)

    df["BB_Std"] = df["Close"].rolling(window=20).std() #Bollinger Bands Standard Deviation
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"] #Bollinger Bands Width

    #Volume based indicators
    df['OBV']=(np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum() #On Balance Volume (OBV)

    df["Vol_MA_10"] = df['Volume'].rolling(window=10).mean() #10-day moving average of volume
    df["Vol_Ratio"] = df["Volume"]/df["Vol_MA_10"]

    #price momentum  + acceleration
    df["Price_Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Acceleration"] = df["Price_Momentum_10"]-df["Price_Momentum_10"].shift(1)

    #zscore normalized momentum
    df["ZMomentum"] = (df["Price_Momentum_10"] - df["Price_Momentum_10"].rolling(20).mean()) / df["Price_Momentum_10"].rolling(20).std()


    #RSI calc (14 day interval)
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)

    avg_gain=gain.rolling(window=14).mean()
    avg_loss=loss.rolling(window=14).mean()

    #'relative strength' calculation
    rs = avg_gain/(avg_loss + 1e-9) 
    df['RSI'] = 100 - (100 / (1+rs)) #RS *INDICATIOR* VALUATION

    #RSI Momentum (delta)
    df["RSI_Delta"] = df["RSI"].diff()

    #lag features

    for lag in [1,3,5]:
        df[f"Return_Lag{lag}"] = df["Return"].shift(lag)
        if "RSI" in df.columns:
            df[f"RSI_Lag_{lag}"] = df["RSI"].shift(lag)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


    


#3 "EVENT LABELING" LOGIC
#BINARY CLASSIFICATION ON BASIS OF (PREDICTED) FUTURE RETURNS
#--Each row labeled as followed: (0==NORMAL (ELSE), 1==CRASH ) 
# + CONFIDENCE PROBABLITY VALUATION (LOOK INTO A LIL BIT)
def label_events(df, crash_threshold=-0.03, spike_threshold=0.03):
    df=df.copy()
    df["Future_Close"]=df['Close'].shift(-1)
    df["Future_Return"]=(df["Future_Close"]-df['Close'])/df['Close']
    df=df.dropna(subset=["Future_Return"])

    df['Crash']=(df["Future_Return"]<crash_threshold).astype(int)
    df['Spike']=(df["Future_Return"]>spike_threshold).astype(int)

    df['Event']=np.select(
        [df['Crash'] == 1, df['Spike'] == 1],
        [1,2], #1 == 'Crash', 2 == 'Spike'
        default=0 #0 == 'Normal'
    )

    return df

#4. Balance dataset
#-DEALS WITH IMBALANCED DATASETS
def balance_dataset(X,y):
    data=pd.concat([X,y], axis=1)
    crash=data[data["Crash"]==1]
    normal=data[data["Crash"]==0]

    #upsampling minorty class (crash data)
    crash_upsampled=resample(crash, replace=True, n_samples=len(normal), random_state=42)
    balanced=pd.concat([normal, crash_upsampled])
    balanced=balanced.sample(frac=1,random_state=42) #shuffle set
    
    return balanced.drop('Crash', axis=1), balanced['Crash']




#5. ML MODEL ARCHETECHURE (BASED ON RANDOM FOREST MODEL)
#-RECURSIVE SELF TRAINING OF ML MODEL
#-- WILL UTILIZE "RANDOM FOREST" STYLED ML MODEL, BASED OFF OF THESE EXTRACTED VALUATIONS/EVERCHANGING DATASET VALUATIONS
#---RANDOM FOREST MODEL: USED FOR INTERPRETABILITLY/ROBUSTNESS OF OVERALL ML ALGORITHM AND ARCHITECHTURE
def train_model(df, features=None, target="Event"):  #in theory, trains our model on above extractions
    if features is None:
        features = get_feature_list()
    #selection of feature and target (X and y variables respectively) from DataFrame
    X=df[features] #features inputted to be used to train our model below
    y=df[target] #deals w/ output labels (0=='normal, 1=='crash", 2=='spike')
    #X,y=balance_dataset(X,y) #balances dataset to deal with imbalanced classes

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting of dat ainto training/test subsets (80/20 split in this case)
    model=RandomForestClassifier(n_estimators=100, random_state=42) #imitilaization of RF classifier, in this case utilizing 100 trees.
    model.fit(X_train, y_train) #train model w/ training data

    y_pred=model.predict(X_test) #prediction crash labels on test set

    #display performance metrics
    print('\nModel Performance Metric Valuations:')
    print('Accuracy')
    print('Classification Report:\n', classification_report(y_test, y_pred))

    #saving of trained model above for future use.
    joblib.dump(model, "market_crash_model.pkl") #Saves our pre trained model (ideally)
    return model


def in_human_speak(prediction, crash_conf, spike_conf):
    #CRASH
    if prediction==1:
        print(f"\n🚨 WARNING!: Market Crash Imminent! Confidence: {crash_conf*100:.1f}%")
        if crash_conf < 0.2:
            print(f"\n✅ MARKET APPEARS STABLE! Model confidence of crash probablitiy very low (confidence: {crash_conf*100:.1f}%)")
        elif crash_conf  < 0.5:
            print(f"\n ⚠️ CAUTION ADVISED!: Model predicts {crash_conf*100:.1f}% chance of crash! Stay aware of market conditions")
        else:
            print(f"\n🔍 MIXED SIGNALS: while model predicts that crash is unlikely, the model shows moderate confidence ({crash_conf*100:.1f}%). Stay vigilant!")
    #SPIKE
    elif prediction == 2:
        if spike_conf > 0.8:
            print(f'\n📈STRONG BUY SIGNAL!: High confidence in spike probability detected! ({spike_conf * 100:.1f}% confidence)')
        elif spike_conf > 0.5:
            print(f"\n✅POSSIBLE UPWARD TREND!: Moderate likelihood of market rally detected ({spike_conf * 100:.1f}% confidence)")
        else:
            print(f"\n🔍 Mild upward movement expected! ({spike_conf * 100:.1f}% confidence)")
    #NEUTRAL
    else:
        if crash_conf > 0.4:
            print(f"\n⚠️NEUTRAL CONSENSUS: however slight crash indicators detected ({crash_conf * 100:.1f}% confidence)")
        elif spike_conf > 0.4:
            print(f"\n⚠️NEUTRAL CONSENSUS: But possible emergence of upward trend present ({crash_conf * 100:.1f}% confidence)")
        else:
            print(f"\n✅MARKET APPEARS STABLE: no significant crash/spike behaviour detected! ({crash_conf * 100:.1f}% confidence)")

#6. LIVE PREDICTION PIPELINE
#6.1: 
#-WILL INCLUDE ACCOMPANYING CUMULATIVE CONFIDENCE SCORES, AS DISTILLED FROM ABOVE PROCESSES
#--MAITENCNCE OF A LIVE CONSISTENT LOG IMPORTANT HERE, AS EVEN IF NO CRASH IS PREDICTED IS STILL CRITICAL COMPONTNET OF GENERATIING A CUMULATIVE DAILY CONFIDENCE TREND VISUALIZATION
def live_predict(df, model_path="market_crash_model.pkl"):
    if not os.path.exists(model_path):
        print('Model file not found')
        return None
    
    model = joblib.load(model_path)

    try:
        features = get_feature_list()
        latest_row = df[features].iloc[-5:].mean().to_frame().T

        prediction = model.predict(latest_row[features])[0]
        class_probs = model.predict_proba(latest_row[features])[0]

        log_prediction_to_file(prediction, class_probs)

        # Human-readable output
        print(f"\nLive Prediction: {prediction}")
        print("Class Probabilities:")
        print(f"Normal: {class_probs[0]*100:.2f}%")
        print(f"Crash:  {class_probs[1]*100:.2f}%")
        print(f"Spike:  {class_probs[2]*100:.2f}%")

        crash_confidence = class_probs[list(model.classes_).index(1)] if 1 in model.classes_ else 0
        spike_confidence = class_probs[list(model.classes_).index(2)] if 2 in model.classes_ else 0

        print(f'Live Prediction: {"CRASH" if prediction == 1 else "SPIKE" if prediction == 2 else "NORMAL"} | Crash Confidence: {crash_confidence:.2f} | Spike Confidence: {spike_confidence:.2f}')

        in_human_speak(prediction, crash_confidence, spike_confidence)
        return prediction, crash_confidence, spike_confidence

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None
 


#6.2: (OPTIONAL, FOR ACCURACY SAKE)
# RETRAIN ML MODEL MONTHLY WITH UPDATED DATASET VALUATIONS 
# --THIS IN THEORY WILL HELP FOR OUR ML MODEL TO ADAPT TO EVER CHANGING MARKET BEHAVIOUR + MAINTAIN A LAYER OF PREDICTION ACCURACY

def retrain_model_monthly(df, features=None, target='Crash'): 
    if features is None:
        features = get_feature_list()
    print("Retraining model with updated data figures...")
    model = train_model(df, features=features, target="Event")
    print("Model retraining successful!")
    return model

#7. (TBD) DATA VISULAIZATION
#-WILL INCLUDE A GRAPHICAL VISUALIZATION OF PREDICTED VS. REAL TIME VALUATIONS
def visualize_data(df, save_path='graphs/daily_plot.png', show=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) #ensures folder exsists
    plt.figure(figsize=(14,7))
    plt.plot(df.index, df['Close'], label="Close Price", alpha=0.6)
    plt.plot(df.index, df["MA_20"], label="{20-Period Moving Average", linestyle="--", alpha=0.8)

    #highlight market crashes
    if "Crash" in df.columns:
        crash_points = df[df["Crash"]==1]
        plt.scatter(crash_points.index, crash_points['Close'], color='red', label="Predicted Market Crashes", zorder=5, marker="v")

    #highlight market spikes
    if "Spike" in df.columns:
        spike_points = df[df["Spike"]==1]
        plt.scatter(spike_points.index, spike_points['Close'], color='green', label="Predicted Market Spikes", zorder=5 , marker="^")

    #PLOT FORMATTING
    plt.title("Stock Pricing w/ Moving Averages + Crash/Spike Events")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    #ENSURE OUTPUT DIR IS PRESENT
    os.makedirs(os.path.dirname(save_path), exist_ok=True) #maybe remove this? idk

    #save figure
    plt.savefig(save_path)
    print(f'[Graph] Saved plot to {save_path}')

    #SHOW ONLY IF SHOW=TRUE
    if show:
        plt.show() #infinite chocopoints for meeeeeee x)
    plt.close() #frees up unneeded memory this way

#8. CONDIDENCE TREND VISUALIZER
def plot_confidence_trend(log_file="daily_predictions.csv", show=True):
    df = pd.read_csv(log_file)
    os.makedirs('graphs', exist_ok=True)


    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    plt.figure(figsize=(10, 6))
    df[["crash_prob", "spike_prob"]].plot(ax=plt.gca())
    plt.title("Crash/Spike Confidence Over Time")
    plt.ylabel("Probability (%)")
    plt.xlabel("Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#9. MAIN PIPELINE
#-MAIN FUNCTIONALITY OF PROGRAM
#--WILL INCLUDE ALL ABOVE FUNCTIONS IN A SEQUENTIAL ORDER
#---WILL ALSO INCLUDE A MAIN FUNCTIONALITY FOR USER TO RUN PROGRAM
if __name__ == '__main__':
    print ("DEBUG: starting program...")
    df=fetch_ohlcv(symbol="SPY", api_key = api_key, outputsize="full")

    if df is not None:
        df = calculate_technical_indicators(df)
        df=label_events(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        model=train_model(df, target='Event')
        live_predict(df) 

    else:
        print("ERROR: Failed to fetch data")

#10. PREDICTION LOG CLEANER
def clean_predicton_log(log_file="prediction_log.txt"):
    df=pd.read_csv(
        log_file,
        names=["Timestamp", 'Prediction', 'Crash_Conf', "Spike_Conf"],
        parse_dates=["Timestamp"],
        dtype={"Prediction": str, "Crash_Conf": str, "Spike_Conf": str}, #loads figures as "str" 
        on_bad_lines='skip', 
    )
    
    #   #Keep only rowsnwhere conf valuations are valid floats
    df = df[pd.to_numeric(df["Crash_Conf"], errors='coerce').notnull()]
    df = df[pd.to_numeric(df["Spike_Conf"], errors="coerce").notnull()]
    df["Crash_Conf"]=df["Crash_Conf"].astype(float)
    df["Spike_Conf"]=df["Spike_Conf"].astype(float)

    #removal of duplicates with nearly iodentical timestamps
    df = df.drop_duplicates(subset="Timestamp") 
    df.to_csv(log_file, index=False, header=False)
    print(f"[Log Cleaner] Cleaned log file {log_file} of duplicates and NaN values")



#11: DAILY SCHEDULER FUNCTIONALITY
#-WILL INCLUDE A DAILY SCHEDULER FUNCTIONALITY TO RUN THE PROGRAM ON A DAILY BASIS
#--WILL INCLUDE A FUNCTIONALITY TO RUN THE PROGRAM ONCE, THEN SCHEDULE IT TO RUN DAILY
#------DAILY SCHEDULER FUNCTION--------#
def daily_job():
    print('[Scheduler] Executing daily market prediction...')

    df=fetch_ohlcv(symbol="SPY", api_key=api_key, outputsize="full")
    if df is not None:
        df=calculate_technical_indicators(df)
        df=label_events(df)
        df=df.replace([np.inf, -np.inf], np.nan).dropna()
        model=train_model(df, target='Event')
        live_predict(df)
        show_combined_dashboard(df) #will handle both graphs anyway
        clean_predicton_log()
    else:
        print("ERROR: Failed to fetch data")

#-----START DAILY SCHEDULER-----#
def start_scheduler():
    #inintial predicitons
    print("[scheduler] Running initial prediction...")
    daily_job() #runs once immediately
    #schedules job for 6pm daily
    schedule.every().day.at('18:00').do(daily_job)
    print('[Scheduler] Scheduled daily_job for 6:00pm')
    print("scheuduler initiatied, now waiting for jobs...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"ERROR: {e}")
            #time.sleep(60)

#-----ENTRY POINT FOR SCHEDULER-----#

def run_once_then_schedule():
    daily_job()
    schedule.every().day.at('18:00').do(daily_job)
    print('[Scheduler] Scheduled daily_job for 6:00pm')
    print('Press Ctrl+C to exit the scheduler')

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Scehduler terminated by user")
        print("\nExiting program...")
        exit(0)

    while True:
        schedule.run_pending()
        time.sleep(60)

#12. Combined Dashboard Functionality
def show_combined_dashboard(df, log_file="prediction_log.txt"):
    #load pred log
    os.makedirs("graphs", exist_ok=True)  # ✅ This ensures 'graphs/' exists



    try:
        log_df=pd.read_csv(
            log_file,
            names=["Timestamp", 'Prediction', 'Crash_Conf', "Spike_Conf"],
            parse_dates=["Timestamp"],
        )
        log_df=log_df.dropna(subset=["Crash_Conf", "Spike_Conf"])
        log_df["Crash_Conf"]=log_df["Crash_Conf"].astype(float)
        log_df["Spike_Conf"]=log_df["Spike_Conf"].astype(float)
    except FileNotFoundError:
        print(f"[Error]: Log file {log_file} not found!")
        return
    
    #set up fig with 2 subplots
    fig, (ax1,ax2)=plt.subplots(2,1,figsize=(14,7), constrained_layout=True) #2r,1c #play around with figsize dimentions!

    #top: stock prices
    ax1.plot(df.index,df['Close'], label="Close Price",alpha=0.7)
    ax1.plot(df.index,df["MA_20"],label="20 DAY MOVING AVERAGE",linestyle="--",alpha=0.8)

    if "Crash" in df.columns:
        crash_points=df[df["Crash"]==1]
        ax1.scatter(crash_points.index, crash_points['Close'],color="red",label="Predicted Crashes", marker="v")
    if "Spike" in df.columns:
        spike_points=df[df["Spike"]==1]
        ax1.scatter(spike_points.index, spike_points['Close'],color="green",label="Predicted Spikes", marker="^")

    ax1.set_title("Stock Prices + Crash Spike Events - With Accompnaying Moving Averages")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    #bottom: confidence trend
    ax2.plot(log_df["Timestamp"], log_df["Crash_Conf"], label="Crash Confidence", color='red', linewidth=2)
    ax2.plot(log_df["Timestamp"], log_df["Spike_Conf"], label="Spike Confidence", color='green', linewidth=2)

    ax2.set_title("Market Crash/Spike Confidence Trend Valuations Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("confidence level")
    ax2.set_ylim(0,1.05)
    ax2.grid(True)
    ax2.legend()
    
    
    # Save directory check and figure save
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/combined_dashboard_{pd.Timestamp.now().date()}.png"
    plt.savefig(filename)
    print(f"[✅] Saved plot to {filename}")
    plt.show()


if __name__ == '__main__':
    run_once_then_schedule()




#for project
#-run program daily
#compare real vs predicted figures side by side, map out trends
#***add a "in human speak" function, just to make valuations less abstract and numerical***

#TO DO:
#-MAKE SURE VALUATIONS AND PREDICTED GET APPENDED TO .TXT FILE ON CHRONOLOGICAL BASIS. MOIDEL IS GREAT FOR PAST DATA ESTIMATIONS, BUT TO TRAIN THE MODEL PRESENT VALUATIONS ARE NECESSARY
#--RETRAIN MODEL ON WEEKLY/MONTHLY BASIS (USE CRON/SCHEDULER LIBRARY FOR THIS!)
#--EXPANSION OF FEATURE ENGINEERING FUNCTION: INCLUDE METRICS LIKE MACD, BOLLINGER BANDS, EMA, VALUME BASED METRICS ETC.
#--BUILD DAILY SCHEDULER TO MAKE THIS HAPPEN

#-********BUILD IN "SPIKE" PREDICTION FUNCTION: WOULD BE AS SIMPLE AS INVERTING THE CURRENT LOGICAL ORDER (AS IN, SPIKE IMMINENT IF FUTURE RETURN<=3%. WOULD BE EASY TO ACHIEVE!)\
# ------ (IF PREDICTED SPIKE, THEN BUY, ELSE SELL)

#additional features to add (not for project per se but just becasue why not)
#-add a feature to compare predicted vs. real time valuations
#plot: volatility spikes, RSI over time
#crash confidence trend over time (based on predioction log)


