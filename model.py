import numpy as np
import pandas as pd

#for data visualization
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('plant(IBM - Z).csv')

# print(df.isna().sum())

df.P = df.P.fillna(value = 0)
df.K = df.K.fillna(value = 0)



df.temperature = df.temperature.fillna(value = df.temperature.mean())
df.humidity = df.humidity.fillna(value = df.humidity.mean())
df.ph = df.ph.fillna(value = df.ph.mean())
df.rainfall = df.rainfall.fillna(value = df.rainfall.mean())
df.label = df.label.fillna(value = df.label.mode()[0])


categor_condn = [(df['rainfall'] <= 150),
                  (df['rainfall'] > 250)]

rating = ['low','high']
df['Water Usage'] = np.select(categor_condn,rating,default = 'medium')

# print(df['Water Usage'].value_counts())


# print(df)
from sklearn.linear_model import LinearRegression as lm

# print(df.isna().sum())

label_encoder = LabelEncoder()
df['Water Usage'] = label_encoder.fit_transform(df['Water Usage'])
# df['label'] = label_encoder.fit_transform(df['label'])

# print(df.head())

# # Applying get dummies function on categorical column
x = df.drop('label',axis = 1)
y = df['label']

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.30 , random_state = 10)

print("Dimension of x_train :",x_train.shape)
print("Dimension of x_test :",x_test.shape)
print("Dimension of y_train :",y_train.shape)
print("Dimension of y_test :",y_test.shape)

from sklearn.ensemble import RandomForestClassifier

# logmodel = lm()
# logmodel.fit(x_train.values,y_train.values)
rfc = RandomForestClassifier(n_estimators = 500,criterion = "entropy")
rfc.fit(x_train.values,y_train.values)
# predict_r = rfc.predict(x_test)

pickle.dump(rfc , open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[61,38,55,76,52,28,180,2]]))

prediction = rfc.predict((np.array([[61,38,55,76,52,28,180,2]])))
print("The suggested crop is : ",prediction) 

prediction = rfc.predict((np.array([[90,42,43,20.87,82,6.5,203,2]])))
print("The suggested crop is : ",prediction) 
