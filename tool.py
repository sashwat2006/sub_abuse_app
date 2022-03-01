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
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

path = "https://raw.githubusercontent.com/sashwat2006/EmpaticaProject/main/FINALDATASET2.csv"
data = pd.read_csv(path)
data = data[["BVP","TEMP","EDA","FINALFLAG"]]
data.head()

def app():

	features = data[["TEMP","BVP","EDA"]]
	target = data["FINALFLAG"]

	x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

	st.title("Enter your health values for prediction")
	tempValue = st.slider("Temperature Values",float(data["TEMP"].min()),float(data["TEMP"].max()))
	bvpValue = st.slider("BVP Values",float(data["BVP"].min()),float(data["BVP"].max()))
	edaValue = st.slider("EDA Values",float(data["EDA"].min()),float(data["EDA"].max()))

	def prediction(model,temp,bvp,eda):
		finalFlag = model.predict([[temp,bvp,eda]])
		finalFlag = finalFlag[0]
		return finalFlag

	model_select = st.selectbox("Choose your classifier",("Random Forest","SVM","Logistic Regression"))
	classification = st.button("Perform Prediction",)

	if model_select == "SVM":
		st.sidebar.subheader("Model HyperParameters")
		c = st.sidebar.number_input("C",1,100,step=1)
		gamma = st.sidebar.number_input("Gamma",1,100,step=1)
		kernel_input = st.sidebar.radio("kernel",("linear","rbf","poly"))
		if classification:
			st.subheader("Support Vector Machine")
			svc_model = SVC(C=c,gamma=gamma,kernel=kernel_input)
			svc_model.fit(x_train,y_train)
			pred = svc_model.predict(x_test)
			acc = svc_model.score(x_test,y_test)
			result = prediction(svc_model,tempValue,bvpValue,edaValue)
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
			col3, col4 = st.columns(2)
			with col3:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_confusion_matrix(svc_model,x_test,y_test,ax=ax)
				st.pyplot()
			with col4:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_roc_curve(svc_model,x_test,y_test,ax=ax)
				st.pyplot()

			st.text('Model Report:\n ' + classification_report(y_test, pred))


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
			result = prediction(rf_model,tempValue,bvpValue,edaValue)
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
			col3, col4 = st.columns(2)
			with col3:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_confusion_matrix(rf_model,x_test,y_test,ax=ax)
				st.pyplot()
			with col4:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_roc_curve(rf_model,x_test,y_test,ax=ax)
				st.pyplot()

			st.text('Model Report:\n ' + classification_report(y_test, pred))

			mesh_size = 1
			x_min, x_max = features.TEMP.min(), features.TEMP.max()
			y_min, y_max = features.BVP.min(), features.BVP.max()
			z_min, z_max = features.EDA.min(), features.EDA.max()
			x_range = np.arange(x_min, x_max, mesh_size)
			y_range = np.arange(y_min, y_max, mesh_size)
			z_range = np.arange(z_min, z_max, mesh_size)
			fig = px.scatter_3d(data, x='TEMP', y='BVP', z='FINALFLAG')
			fig.update_layout(scene=dict(xaxis = dict(nticks=4, range=[25, 45]),yaxis = dict(nticks=4, range=[-500,500],),zaxis = dict(nticks=4, range=[-0.2,1.2],),),width=700,margin=dict(r=20, l=10, b=10, t=10))
			st.plotly_chart(fig)


	if model_select == "Logistic Regression":
	    st.sidebar.subheader("Model HyperParameters")
	    c = st.sidebar.number_input("C",1,100,step=1)
	    max_iter = st.sidebar.number_input("Max Iterations",100,1000,step=10)
	    if classification:
	        st.subheader("Logistic Regression Model")
	        lr_model = LogisticRegression(C=c,max_iter=max_iter)
	        lr_model.fit(x_train,y_train)
	        pred = lr_model.predict(x_test)
	        acc = lr_model.score(x_test,y_test)
	        result = prediction(lr_model,tempValue,bvpValue,edaValue)
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
	        col3, col4 = st.columns(2)
	        with col3:
	        	fig, ax = plt.subplots(figsize=(5,5))
	        	plot_confusion_matrix(lr_model,x_test,y_test,ax=ax)
	        	st.pyplot()
	        with col4:
	        	fig, ax = plt.subplots(figsize=(5,5))
	        	plot_roc_curve(lr_model,x_test,y_test,ax=ax)
	        	st.pyplot()

	        st.text('Model Report:\n ' + classification_report(y_test, pred))

	        mesh_size = 1
	        x_min, x_max = features.TEMP.min(), features.TEMP.max()
	        y_min, y_max = features.BVP.min(), features.BVP.max()
	       	z_min, z_max = data["FINALFLAG"].min(), data["FINALFLAG"].max()
	        x_range = np.arange(x_min, x_max, mesh_size)
	        y_range = np.arange(y_min, y_max, mesh_size)
	        z_range = np.arange(z_min, z_max, mesh_size)
	        fig = px.scatter_3d(data, x='TEMP', y='BVP', z='FINALFLAG')
	        fig.update_layout(scene=dict(xaxis = dict(nticks=4, range=[25, 45]),yaxis = dict(nticks=4, range=[-500,500],),zaxis = dict(nticks=4, range=[-0.2,1.2],),),width=700,margin=dict(r=20, l=10, b=10, t=10))
	        st.plotly_chart(fig)