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
import toolv2
import tool
import toolv3
import data1
import plotly.express as px
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

path = "https://raw.githubusercontent.com/sashwat2006/EmpaticaProject/main/FINALDATASET2.csv"
data = pd.read_csv(path)
data = data[["BVP","TEMP","EDA","Gender","Race","Education Level","Employment","FINALFLAG"]]
data.head()

st.title("SAP-ML: Novel Substance Abuse Prediction Platform")

pages_dict = {"Prediction Tool":toolv2,"Data Visualization Page":data1}
st.sidebar.title("Navigation")
user_choice = st.sidebar.radio("Go To",tuple(pages_dict.keys()))
if user_choice == "Prediction Tool":
	toolv2.app(data)
elif user_choice == "Data Visualization Page":
	data1.app()




