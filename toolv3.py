import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

path = "https://raw.githubusercontent.com/sashwat2006/EmpaticaProject/main/FINALDATASET2.csv"
data = pd.read_csv(path)
data = data[["BVP","TEMP","EDA","Gender","FINALFLAG"]]
data.head()

def app():

	features = data[["BVP","TEMP","EDA","Gender"]]
	target = data["FINALFLAG"]

	data["Gender"] = data["Gender"].map({"Male":0,"Female":1})

	x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

	st.title("Enter your health values for prediction")
	tempValue = st.slider("Temperature Values",float(data["TEMP"].min()),float(data["TEMP"].max()))
	bvpValue = st.slider("BVP Values",float(data["BVP"].min()),float(data["BVP"].max()))
	edaValue = st.slider("EDA Values",float(data["EDA"].min()),float(data["EDA"].max()))
	gender = st.selectbox("Select your gender",("Male","Female"))

	if(gender=="Male"):
		gender=0
	elif(gender=="Female"):
		gender=1

	st.write("gender is ",gender)


	model_select = st.selectbox("Choose your classifier",("Random Forest","SVM","Logistic Regression"))
	classification = st.button("Perform Prediction",)


	if model_select == "Random Forest":
		st.sidebar.subheader("Model HyperParameters")
		n_estimators = st.sidebar.number_input("N Estimators",100,1000,step=10)
		max_depth = st.sidebar.number_input("Max Depth",1,100,step=1)
		if classification: 
			st.subheader("Random Forest")
			rf_model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
			rf_model.fit(x_train,y_train)
			pred = rf_model.predict(x_test)
			acc = rf_model.score(x_test,y_test)
			result = rf_model.predict([[bvpValue,tempValue,edaValue,gender]])
			col1, col2 = st.columns(2)
			if(result==0.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk is ","HIGH RISK","-")
				with col2:
					st.metric("Prediction Accuracy", acc)
			if(result==1.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk","NO RISK","+")
				with col2:
					st.metric("Prediction Accuracy", acc)