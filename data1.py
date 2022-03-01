import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)

path = "https://raw.githubusercontent.com/sashwat2006/EmpaticaProject/main/FINALDATASET2.csv"
data = pd.read_csv(path)
data = data[["BVP","TEMP","EDA","Gender","Race","Education Level","Employment","FINALFLAG"]]
data.head()

def app():
	st.title("Welcome to the Data Visualization Page")

	st.header("View Data")
	with st.expander("Click this to show first 5 rows and last 5 rows of the dataset >> (0.0 shows High Risk and 1.0 shows No Risk)"):
		st.table(data.iloc[np.r_[0:5, -5:0]])
	st.header("Description of data")
	with st.expander("Click this to show the description of the Dataset"):
		st.table(data.describe())

	#col1, col2 = st.columns(2)
	#with col1:
	#	st.checkbox("Show column names")
	#	st.table(list(data.columns))
	#with col2:
	#	selection = st.selectbox("Select column to view data",tuple(data.columns))
	#	st.write(data[selection])

	multi_select = st.multiselect("Choose plot type",("BoxPlot","Countplot","Correlation Heatmap"))

	if "Countplot" in multi_select:
		st.subheader("Count Plot")
		var1 = st.multiselect("Select columns for countplot",("Gender","Race","Education Level","Employment"))
		st.set_option('deprecation.showPyplotGlobalUse', False)
		for i in var1:
			st.subheader(f"Countplot of {i}")
			plt.figure(figsize=(10,10))
			sns.countplot(x=data[i],hue=data["FINALFLAG"])
			st.pyplot()





	if "BoxPlot" in multi_select:	
		st.subheader("Box Plot")
		var2 = st.multiselect("Select columns for boxplot",('BVP', 'TEMP', 'EDA'))
		st.set_option('deprecation.showPyplotGlobalUse', False)
		for i in var2:
			st.subheader(f"Box Plot of {i}")
			plt.figure(figsize=(12,7))
			sns.boxplot(data[i])
			st.pyplot()

	if "Correlation Heatmap" in multi_select:
		st.subheader("Heat Map")
		plt.figure(figsize=(10,5))
		sns.heatmap(data.corr(),annot=True)
		st.pyplot()

	selection = st.sidebar.multiselect("Select X Axis Values for Risk Threshold",('TEMP', 'BVP', 'EDA'))
	for i in selection:
		if(i=="TEMP"):
			st.subheader(f"Risk Threshold - {i}")
			plt.figure(figsize=(10,5))
			plt.scatter(data[i],data["FINALFLAG"])
			plt.xlabel(f"{i} ℃" )
			plt.ylabel("Risk")
			st.pyplot()

		if(i=="BVP"):
			st.subheader(f"Risk Threshold - {i}")
			plt.figure(figsize=(10,5))
			plt.scatter(data[i],data["FINALFLAG"])
			plt.xlabel(f"{i}" )
			plt.ylabel("Risk")
			st.pyplot()

		if(i=="EDA"):
			st.subheader(f"Risk Threshold - {i}")
			plt.figure(figsize=(10,5))
			plt.scatter(data[i],data["FINALFLAG"])
			plt.xlabel(f"{i} μs" )
			plt.ylabel("Risk")
			st.pyplot()
	

	#var2 = st.multiselect("Select columns for scatter plot",('BVP', 'TEMP', 'EDA'))

