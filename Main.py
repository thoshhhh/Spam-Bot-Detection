import pandas as pd
import numpy as np
import pickle
import os
from Attention import attention
from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.ensemble import RandomForestClassifier

#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory

app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv("Dataset/twitter_spammer.csv")
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset.dropna(inplace = True)
Y = dataset[['spammer']].mean(axis=1)
Y = Y.ravel()
dataset.drop(['spammer'], axis = 1,inplace=True)

X = dataset.values
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

def getModel():
    #Create GNN model to detect fault from all services
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    gnn_model = Sequential()
    gnn_model.add(GraphCNN(128, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(5e-4)))
    gnn_model.add(GraphCNN(64, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(5e-4)))
    gnn_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(5e-4)))
    gnn_model.add(attention(return_sequences=True,name='attention'))#adding squeeze attention model to make improved Yolov5
    gnn_model.add(Dense(units = 256, activation = 'relu'))
    gnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    gnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    gnn_model.load_weights("model/gnn_weights.hdf5")
    return gnn_model

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('UserScreen.html', msg="<font size=3 color=blue>Welcome "+user+"</font>")
        else:
            return render_template('UserLogin.html', msg="<font size=3 color=red>Invalid login details</font>")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global scaler, dataset, labels, rf
        testData = pd.read_csv("Dataset/testData.csv")#load test data
        columns = testData.columns
        data = testData.values
        labels = ["Genuine", "Spammer"]
        testData = testData.values    
        testData = scaler.transform(testData)#normalize test data
        gnn_model = getModel()
        output = '<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</font></th>'
        output += '<th><font size="3" color="blue">Predicted Account Type</font></th></tr>'
        predict = rf.predict(testData).astype(int)
        print(predict)
        for i in range(len(predict)):
            output += "<tr>"
            for k in range(len(data[i])):
                output += '<td><font size="3" color="black">'+str(round(data[i,k],3))+'</font></td>'
            if labels[predict[i]] == "Genuine":
                output += '<td><font size="4" color="green">'+labels[predict[i]]+'</font></td>'
            else:
                output += '<td><font size="4" color="red">'+labels[predict[i]]+'</font></td>'
        '''
        for i in range(len(testData)): #loop each test
            temp = []
            temp.append(testData[i])
            temp = np.asarray(temp)
            predict = gnn_model.predict(temp, batch_size=1)#apply GNN model on test data to predict spammer or genuine account
            predict = np.argmax(predict)
            print(predict)
            output += "<tr>"
            for k in range(len(data[i])):
                output += '<td><font size="3" color="black">'+str(round(data[i,k],3))+'</font></td>'
            if labels[predict] == "Genuine":
                output += '<td><font size="4" color="green">'+labels[predict]+'</font></td>'
            else:
                output += '<td><font size="4" color="red">'+labels[predict]+'</font></td>'
        '''        
        output += "</table><br/><br/><br/><br/>"             
        return render_template('UserScreen.html', msg=output)
        

if __name__ == '__main__':
    app.run()    
