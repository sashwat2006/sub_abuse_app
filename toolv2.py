import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


st.set_option('deprecation.showPyplotGlobalUse', False)

path = "https://raw.githubusercontent.com/sashwat2006/EmpaticaProject/main/FINALDATASET2.csv"

data = pd.read_csv(path)
data = data[["BVP","TEMP","EDA","Gender","Race","Education Level","Employment","FINALFLAG"]]
raw_data = pd.read_csv(path)
raw_data = data[["BVP","TEMP","EDA","Gender","Race","Education Level","Employment","FINALFLAG"]]
data.head()

def app(data):

	
	data["Gender"] = data["Gender"].map({"Male":0,"Female":1})
	data["Race"] = data["Race"].map({"White":0,"African American":1,"Asian":2,"Latino":3,"American Indian":4,"Other":5})
	data["Education Level"] = data["Education Level"].map({"High School Graduate":0,"High School Incomplete":1})
	data["Employment"] = data["Employment"].map({"Unemployed":0,"Part Time":1,"Full Time":2})



	st.title("Enter your health values for prediction")
	tempValue = st.slider("Temperature Values",float(data["TEMP"].min()),float(data["TEMP"].max()))
	bvpValue = st.slider("BVP Values",float(data["BVP"].min()),float(data["BVP"].max()))
	edaValue = st.slider("EDA Values",float(data["EDA"].min()),float(data["EDA"].max()))
	race = st.selectbox("Select your race",("White","Asian","American Indian","African American","Latino","Other"))
	gender = st.selectbox("Select your gender",("Male","Female"))
	education = st.selectbox("Select your education level",("High School Graduate","High School Incomplete"))
	employment = st.selectbox("Select your current employment status",("Unemployed","Part Time","Full Time"))


	tempValue = (tempValue-data["TEMP"].mean())/data["TEMP"].std()
	bvpValue = (bvpValue-data["BVP"].mean())/data["BVP"].std()
	edaValue = (edaValue-data["EDA"].mean())/data["EDA"].std()

	scaler = StandardScaler()
	df = scaler.fit_transform(data[["TEMP","BVP","EDA"]])
	df = pd.DataFrame(df,columns=["TEMP","BVP","EDA"])
	data = df.join(data[["Gender","Race","Education Level","Employment","FINALFLAG"]])
	features = data[["BVP","TEMP","EDA","Gender","Race","Education Level","Employment"]]
	target = data["FINALFLAG"]


	x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42,stratify=target)



	if(race=="White"):
		race=0
	elif(race=="African American"):
		race=1
	elif(race=="Asian"):
		race=2
	elif(race=="Latino"):
		race=3
	elif(race=="American Indian"):
		race=4
	elif(race=="Other"):
		race=5

	if(gender=="Male"):
		gender=0
	elif(gender=="Female"):
		gender=1

	if(education=="High School Graduate"):
		education=0
	elif(education=="High School Incomplete"):
		education=1

	if(employment=="Unemployed"):
		employment=0
	elif(employment=="Part Time"):
		employment=1
	elif(employment=="Full Time"):
		employment=2

	def prediction(model,bvp,temp,eda,gender,race,education,employment):

		finalFlag = model.predict([[bvp,temp,eda,gender,race,education,employment]])
		finalFlag = finalFlag[0]
		return finalFlag

	model_select = st.selectbox("Choose your classifier",("Random Forest","Logistic Regression","Decision Tree"))
	var1 = st.multiselect("Select columns for countplot - Predictions",("Gender","Race","Education Level","Employment"))
	var2 = st.multiselect("Select columns for countplot - Raw Data",("Gender","Race","Education Level","Employment"))
	classification = st.button("Perform Prediction",)

	if model_select == "Decision Tree":
		st.sidebar.subheader("Model HyperParameters")
		splitter = st.sidebar.radio("split technique",("best","random"))
		max_depth = st.sidebar.number_input("Max Depth",1,100,step=1)
		if classification:
			st.subheader("Decision Tree Classifier")
			d_tree = DecisionTreeClassifier(splitter=splitter,max_depth=max_depth)
			d_tree.fit(x_train,y_train)
			pred = d_tree.predict(x_test)
			acc = d_tree.score(x_test,y_test)
			result = prediction(d_tree,bvpValue,tempValue,edaValue,gender,race,education,employment)
			col1, col2 = st.columns(2)
			if(result==0.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk is ","NO RISK","-")
				with col2:
					st.metric("Prediction Accuracy", acc)
			if(result==1.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk","HIGH RISK","+")
				with col2:
					st.metric("Prediction Accuracy", acc)
			col3, col4 = st.columns(2)
			with col3:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_confusion_matrix(d_tree,x_test,y_test,ax=ax)
				st.pyplot()
			with col4:
				fig, ax = plt.subplots(figsize=(5,5))
				plot_roc_curve(d_tree,x_test,y_test,ax=ax)
				st.pyplot()

			for i in var2:
				st.subheader(f"Count Plot based on True, Raw Data for {i}")
				plt.figure(figsize=(10,10))
				fig = px.histogram(x=x_train[i],color=y_train,barmode="group")
				fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk",legend_title="Risk Type")
				st.plotly_chart(fig)

			for i in var1:
				st.subheader(f"Count Plot based on Model's Predictions for {i}")
				plt.figure(figsize=(10,10))
				fig = px.histogram(x=x_test[i],color=pred,barmode="group")
				fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk", legend_title="Risk Type")
				st.plotly_chart(fig)

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

			result = prediction(rf_model,bvpValue,tempValue,edaValue,gender,race,education,employment)

			
			col1, col2 = st.columns(2)
			if(result==0.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk is ","NO RISK","-")
				with col2:
					st.metric("Prediction Accuracy", acc)
			if(result==1.0):
				with col1:
					st.metric("Predicted Substance Abuse Risk","HIGH RISK","+")
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


			for i in var2:
				st.subheader(f"Count Plot based on True, Raw Data for {i}")
				plt.figure(figsize=(10,10))
				fig = px.histogram(x=x_train[i],color=y_train,barmode="group")
				fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk",legend_title="Risk Type")
				st.plotly_chart(fig)

			for i in var1:
				st.subheader(f"Count Plot based on Model's Predictions for {i}")
				plt.figure(figsize=(10,10))
				fig = px.histogram(x=x_test[i],color=pred,barmode="group")
				fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk", legend_title="Risk Type")
				st.plotly_chart(fig)


			

			st.text('Model Report:\n ' + classification_report(y_test, pred))

			
			

			#mesh_size = 1
			#x_min, x_max = features.TEMP.min(), features.TEMP.max()
			#y_min, y_max = features.BVP.min(), features.BVP.max()
			#z_min, z_max = features.EDA.min(), features.EDA.max()
			#x_range = np.arange(x_min, x_max, mesh_size)
			#y_range = np.arange(y_min, y_max, mesh_size)
			#z_range = np.arange(z_min, z_max, mesh_size)
			#fig = px.scatter_3d(data, x='TEMP', y='BVP', z='FINALFLAG')
			#fig.update_layout(scene=dict(xaxis = dict(nticks=4, range=[25, 45]),yaxis = dict(nticks=4, range=[-500,500],),zaxis = dict(nticks=4, range=[-0.2,1.2],),),width=700,margin=dict(r=20, l=10, b=10, t=10))
			#st.plotly_chart(fig)


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
	        result = prediction(lr_model,bvpValue,tempValue,edaValue,gender,race,education,employment)
	        col1, col2 = st.columns(2)
	        if(result==0.0):
	        	with col1:
	        		st.metric("Predicted Substance Abuse Risk is ","NO RISK","-")
	        	with col2:
	        	    st.metric("Prediction Accuracy", acc)	 

	        if(result==1.0):
	        	with col1:
	        		st.metric("Predicted Substance Abuse Risk","HIGH RISK","+")
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

	        for i in var2:
	        	st.subheader(f"Count Plot based on True, Raw Data for {i}")
	        	plt.figure(figsize=(10,10))
	        	fig = px.histogram(x=x_train[i],color=y_train,barmode="group")
	        	fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk",legend_title="Risk Type")
	        	st.plotly_chart(fig)

	        for i in var1:
	        	st.subheader(f"Count Plot based on Model's Predictions for {i}")
	        	plt.figure(figsize=(10,10))
	        	fig = px.histogram(x=x_test[i],color=pred,barmode="group")
	        	fig.update_layout(xaxis_title=f"{i}",yaxis_title="Risk", legend_title="Risk Type")
	        	st.plotly_chart(fig)

	        st.text('Model Report:\n ' + classification_report(y_test, pred))

	      