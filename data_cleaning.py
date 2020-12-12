import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# place data in as a pandas Dataframe with gender already binary
data = pd.read_csv('training_data.csv', converters={'Gender': lambda x: float(x == 'Female')})

# get rid of 'id' in data
data = data.drop(['id'], axis=1)

# Creates column on data called 'Age_Bins' that captures the ages in 5 year intervals
data['Age_Bins'] = pd.cut(x=data['Age'], bins=[19, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])

# Change Age_Bins to numbers and add to data
le = LabelEncoder()
label_age_bins = le.fit_transform(data["Age_Bins"])
label_age_bins = (label_age_bins.reshape(-1, 1))
data["Numbered_Bins"] = label_age_bins

# One Hot encode Age_Bins
ohe = OneHotEncoder(sparse=False)
encoded_bins = ohe.fit_transform(label_age_bins)
encoded_bins = pd.DataFrame(encoded_bins)
data = data.join(encoded_bins.add_prefix("Age_"))

# Drop unnecessary age columns
data = data.drop(['Age', 'Age_Bins'], axis=1)

# One Hot region codes
ohe = OneHotEncoder(sparse=False)
regions = pd.DataFrame(data["Region_Code"])
regions_enc = pd.DataFrame(ohe.fit_transform(regions))
data = data.join(regions_enc.add_prefix("Region_"))

# drop region code
data = data.drop(['Region_Code'], axis=1)

# Change Vehicle Age Categories to 0, 1, and 2
data['Vehicle_Age'] = data['Vehicle_Age'].replace(['> 2 Years'], 0)
data['Vehicle_Age'] = data['Vehicle_Age'].replace(['1-2 Year'], 1)
data['Vehicle_Age'] = data['Vehicle_Age'].replace(['< 1 Year'], 2)

# One Hot vehicle age
ohe = OneHotEncoder(sparse=False)
vehicle_age = pd.DataFrame(data["Vehicle_Age"])
vehicle_age_enc = pd.DataFrame(ohe.fit_transform(vehicle_age))
data = data.join(vehicle_age_enc.add_prefix("Vehicle_Age_"))

# drop vehicle_age
data = data.drop(['Vehicle_Age'], axis=1)

# binary for vehicle damage
data['Vehicle_Damage'] = data['Vehicle_Damage'].replace(['Yes'], 0)
data['Vehicle_Damage'] = data['Vehicle_Damage'].replace(['No'], 1)

# normalize annual premium
annual = data['Annual_Premium'].values.reshape(-1, 1)
scaler = MinMaxScaler()
annual_scaled = scaler.fit_transform(annual)
Annual_Normalized = pd.DataFrame(annual_scaled)

# Drop old Annual_Premium
data = data.drop(['Annual_Premium'], axis=1)

# Add normalized Annual_Premium
data.insert(6, 'Annual_Premium', Annual_Normalized, True)

# One Hot encode Policy_Sales
ohe = OneHotEncoder(sparse=False)
policy = pd.DataFrame(data["Policy_Sales_Channel"])
policy_enc = pd.DataFrame(ohe.fit_transform(policy))
data = data.join(policy_enc.add_prefix("Policy_"))

# drop policy sales channel
data = data.drop(["Policy_Sales_Channel"], axis=1)

# Standardize Vintage with same method as before
vintage = data['Vintage'].values.reshape(-1, 1)
vintage_scaled = scaler.fit_transform(vintage)
Vintage_Normalized = pd.DataFrame(vintage_scaled)

# drop vintage
data = data.drop(['Vintage'], axis=1)

# Add normalized Vintage
data.insert(5, 'Vintage', Vintage_Normalized, True)

#
# BEGINNING OF MODEL WORK
#

# One Hot vehicle response (should be 2)
ohe = OneHotEncoder(sparse=False)
response = pd.DataFrame(data["Response"])
response_enc = pd.DataFrame(ohe.fit_transform(response))
data = data.join(response_enc.add_prefix("Response_"))

y = np.array(data['Response_0'])

# Get rid of unused Responses
data = data.drop(["Response_0"], axis=1)
data = data.drop(["Response_1"], axis=1)
data = data.drop(["Response"], axis=1)

# Data to be trained
x = np.array(data.values)

# Split data for testing and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Create model
model = Sequential()
model.add(layers.Dense(25, input_dim=231, activation='relu'))
model.add(layers.Dense(50, input_dim=231, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['acc'])

# train data on model
history = model.fit(x_train, y_train, epochs=5, verbose=True, validation_data=(x_test, y_test), batch_size=128)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
