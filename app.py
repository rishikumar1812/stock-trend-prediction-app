# importing the important libary
import pandas as pd
import numpy as np
import pandas_datareader as data
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model

# fetch the data from yahoo finance
import yfinance as yf
import pandas as pd
import datetime as dt

# Define the ticker symbol and date range

start_date = "2018-01-01"
end_date = dt.datetime.now()

st.title('Stock Trend Prediction')

st.subheader('''Use tickers for searches, such as Apple stock (AAPL), Bitcoin stock (BIT-USD)
Here are the top 10 tickers from Yahoo Finance:

1. Apple Inc. (AAPL)
2. Microsoft Corporation (MSFT)
3. Amazon.com Inc. (AMZN)
4. Tesla Inc. (TSLA)
5. Alphabet Inc. (GOOGL)
6. NVIDIA Corporation (NVDA)
7. Meta Platforms Inc. (META)
8. Berkshire Hathaway Inc. (BRK-B)
9. JPMorgan Chase & Co. (JPM)
10. Johnson & Johnson (JNJ)

These tickers can be used for searching financial information on this website.''')


user_input = st.text_input("Enter stock ticker:", "AAPL")
start_date = "2018-01-01"
end_date = dt.datetime.now()

# Fetch data
try:
    data = yf.download(user_input, start=start_date, end=end_date)
    
    # Check if data is empty
    if data.empty:
        st.error("No data found for the given ticker and date range.")
    else:
        st.write(data)
except Exception as e:
    st.error(f"Error fetching data: {e}")

# Create a Pandas DataFrame
df = pd.DataFrame(data)

#Describing data
st.subheader('Data from 2019 to Now')
st.write(df.describe())


#Visulization
st.subheader('Closing Price vs Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA&200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200,'g', label='200MA')
plt.plot(df.Close)
st.pyplot(fig)

#Now to convert training data into range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

#Splitting the data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training)
print(data_testing)

#loading the model
model = load_model('my_model.keras')

#testing part
past_100_days  = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test), np.array(y_test)

#making predictions

y_predicted = model.predict(x_test)
scaler=scaler.scale_
scale_factor = 1/scaler[0]
y_predicted= scale_factor*y_predicted
y_test=scale_factor*y_test




#Final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
