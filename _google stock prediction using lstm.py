#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[42]:


google_stock_data = pd.read_csv('GOOG.csv')
google_stock_data.head()


# In[5]:


google_stock_data.info()


# In[6]:


google_stock_data = google_stock_data[['date','open','close']] # Extracting required columns
google_stock_data['date'] = pd.to_datetime(google_stock_data['date'].apply(lambda x: x.split()[0])) # Selecting only date
google_stock_data.set_index('date',drop=True,inplace=True) # Setting date column as index
google_stock_data.head()


# In[7]:


fg, ax =plt.subplots(1,2,figsize=(20,7))
ax[0].plot(google_stock_data['open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()

ax[1].plot(google_stock_data['close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()

fg.show()


# In[8]:


#Data Pre-Processing
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
google_stock_data[google_stock_data.columns] = MMS.fit_transform(google_stock_data)


# In[9]:


google_stock_data.shape


# In[10]:


training_size = round(len(google_stock_data) * 0.80) # Selecting 80 % for training and 20 % for testing
training_size


# In[11]:


train_data = google_stock_data[:training_size]
test_data  = google_stock_data[training_size:]

train_data.shape, test_data.shape


# In[12]:


# Function to create sequence of data for training and testing

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 0

  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx += 1
  return (np.array(sequences),np.array(labels))


# In[13]:


train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)


# In[14]:


train_seq.shape, train_label.shape, test_seq.shape, test_label.shape


# In[26]:


pip install keras


# In[27]:


#Creating LSTM model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


# In[28]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()


# In[29]:


model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)


# In[30]:


test_predicted = model.predict(test_seq)
test_predicted[:5]


# In[31]:


test_inverse_predicted = MMS.inverse_transform(test_predicted) # Inversing scaling on predicted data
test_inverse_predicted[:5]


# In[36]:


#Visualizing predicted and actual data
#Merging actual and predicted data for better visualization
gs_slic_data = pd.concat([google_stock_data.iloc[-202:].copy(),pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],index=google_stock_data.iloc[-202:].index)], axis=1)


# In[37]:


gs_slic_data[['open','close']] =MMS.inverse_transform(gs_slic_data[['open','close']])#inverse scaling


# In[38]:


gs_slic_data.head()


# In[40]:


gs_slic_data[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('date',size=15)
plt.ylabel('stock price',size=15)
plt.title('actual vs predicted for open price',size=15)
plt.show()


# In[41]:


gs_slic_data[['close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('date',size=15)
plt.ylabel('stock price',size=15)
plt.title('actual vs predicted for close price',size=15)
plt.show()


# In[43]:


#Predicting upcoming 10 days
# Creating a dataframe and adding 10 days to existing index  
gs_slic_data = gs_slic_data.append(pd.DataFrame(columns=gs_slic_data.columns,index=pd.date_range(start=gs_slic_data.index[-1],periods=11, freq='D', closed='right')))


# In[44]:


gs_slic_data['2021-06-09       ':'2021-06-16']


# In[45]:


upcoming_prediction = pd.DataFrame(columns=['open','close'],index=gs_slic_data.index)
upcoming_prediction.index= pd.to_datetime(upcoming_prediction.index)


# In[47]:


curr_seq = test_seq[-1:]

for i in range(-10,0):
    up_pred = model.predict(curr_seq)
    upcoming_prediction.iloc[i]=up_pred
    curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
    curr_seq =curr_seq.reshape(test_seq[-1:].shape)


# In[48]:


upcoming_prediction[['open','close']] = MMS.inverse_transform(upcoming_prediction[['open','close']])


# In[49]:


fg,ax=plt.subplots(figsize=(10,5))
ax.plot(gs_slic_data.loc['2021-04-01':,'open'],label='Current Open Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming Open price prediction',size=15)
ax.legend()
fg.show()


# In[50]:


fg,ax=plt.subplots(figsize=(10,5))
ax.plot(gs_slic_data.loc['2021-04-01':,'close'],label='Current close Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming close price prediction',size=15)
ax.legend()
fg.show()


# In[ ]:




