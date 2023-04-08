import firebase_admin
from firebase_admin import credentials, firestore, db
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Use a service account
try:
    app = firebase_admin.get_app()
except ValueError as e:
    cred = credentials.Certificate("plant2iot-firebase-adminsdk-5wx4h-9c5374ca50.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://plant2iot-default-rtdb.firebaseio.com/'})

################################################ Web App ###########################################################
st.set_page_config(page_title='Analytics App', layout='centered', initial_sidebar_state='auto', page_icon="🌱")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Plant2IoT Analytics App")
st.write("This is a web app for analytics of Plant2IoT project. The data is fetched from the Firebase Realtime Database. The data is then converted to a dataframe and then the data is visualized using Seaborn and Matplotlib. The data is also used for Machine Learning using KMeans Clustering and PCA. And the model is generated and can be saved using Pickle. The model can be used for prediction of the plant's health.")
st.markdown("""
#### **Current data and time:**""")
today = datetime.now()
today = today.strftime("%B %d, %Y %H:%M:%S")
st.write(today)

ref_iot = db.reference('/iot/')
ds = ref_iot.get()
ref_count = db.reference('/count/')
ds2 = ref_count.get()+1
df = pd.DataFrame(ds)

st.markdown("""


""")
st.markdown(
    """
<style>
[data-testid="stMetricLabel"] p{
    font-size: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)

def diff(x):
    return str(df[x].iloc[-1] - df[x].iloc[-2])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="🌡️ Temperature (°C)",
    value=str(df["Temp"].iloc[-1]),
    delta=diff("Temp"),
)

kpi2.metric(
    label="💧 Soil Moisture",
    value= str(df["Moisture"].iloc[-1]),
    delta=diff("Moisture"),
)

kpi3.metric(
    label="🌤️ Light Intensity",
    value=str(df["Light"].iloc[-1]),
    delta=diff("Light"),
)

kpi4.metric(
    label="🌫️ Humidity (%)",
    value=str(df["Hum"].iloc[-1]),
    delta=diff("Hum"),
)
st.markdown("""


""")
if st.button("Fetch Data from Firebase"):
    ref_iot = db.reference('/iot/')
    ds = ref_iot.get()
    ref_count = db.reference('/count/')
    ds2 = ref_count.get()+1
    df = pd.DataFrame(ds)
    st.write("Fetching data from Firebase...")
    st.write("Data fetched successfully!")
    st.write("Total rows:",ds2)
    st.write(df.head())
    st.write("Shape:",df.shape)
    st.write("Describe:",df.describe())

def dowload_file(df):
    b64 = base64.b64encode(df.to_csv().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    return href
st.markdown(dowload_file(df), unsafe_allow_html=True)
    
############################################# Data Visualization ####################################################
st.subheader("Data Visualization")

def plot_analytics():
    sns.set_theme(style="darkgrid")
    st.markdown("""**1.** The following plot shows the **Soil Moisture** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(df.index.values , df["Moisture"],".-g",ms = 7, mfc="k")
    plt.xlabel("Time")
    plt.ylabel("Moisture")
    st.pyplot(fig)
    
    st.markdown("""**2.** The following plot shows the **Temperature** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(df.index.values , df["Temp"],".-r",ms = 7, mfc="g")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    st.pyplot(fig)
    
    st.markdown("""**3.** The following plot shows the **Humidity** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(df.index.values , df["Hum"],".-b",ms = 7, mfc="r")
    plt.xlabel("Time")
    plt.ylabel("Humidity")
    st.pyplot(fig)
    
    st.markdown("""**4.** The following plot shows the **Light** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(df.index.values , df["Light"],".-k",ms = 7, mfc="gray")
    plt.xlabel("Time")
    plt.ylabel("Light Intensity")
    st.pyplot(fig)
    
    st.markdown("""**5.** The Correlation Matrix of the data is shown below.""")
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.write(fig)
    
if st.button("Plot Analytics"):
    plot_analytics()
    st.markdown("""**6.** The latest data is shown below.""")
    st.write(df.tail())
    
########################################################## Machine Learning #########################################
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


st.subheader("Machine Learning")
st.markdown("""
**1.** The following model is used to cluster the unnsupervised data.

**2.** K Means Clustering is used.

**3.** The model is trained on the data collected from the plant.

**4.** The model is then used to predict the condition of the plant.

**5.** Different clusters are formed based on the data.
""")

def model_gen():
    data = df
    inertias = []
    
    pcs = PCA(n_components=2)
    pcs.fit(data)
    datap = pcs.transform(data)

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(datap)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    st.pyplot(fig)
    st.markdown("""
    Elbow method is used to find the optimal number of clusters.
    
    The optimal number of clusters is **3**.
    
    These clusters are {0:"Moist", 1:"Dry", 2:"Healthy"}
    """)
    model = KMeans(n_clusters=3, init='k-means++',random_state=69)
    model.fit(datap)

    dfk = pd.concat([data.reset_index(drop=True), pd.DataFrame(datap, columns=['PC1', 'PC2'])], axis=1)
    dfk['Cluster'] = model.labels_

    st.write("The clusters are shown below.")
    st.write(dfk.head())
    st.markdown("""
    After clustering and analyzing the data, the following clusters are formed.
    A new column is added to the dataframe which shows the condition of the plant.
    """)

    d = {0:"Moist", 1:"Dry", 2:"Healthy"}
    
    dfk["Condition"] = dfk["Cluster"].map(d)
    
    st.table(pd.crosstab(dfk["Cluster"], dfk["Condition"]))
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=dfk, x='PC1', y='PC2', hue='Condition', palette='Set1', ax=ax)
    plt.title('Clusters')
    st.pyplot(fig)
    
    def dowload_model(model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download Trained Model</a>'
        return href
    st.markdown(dowload_model(model), unsafe_allow_html=True)
    
    st.markdown(dowload_file(dfk), unsafe_allow_html=True)

    ######################################
    
    
if st.button("Generate Model"):
    model_gen()

########################################################## Conclusion ################################################
#### Pipeline ####
pipe = Pipeline([('pca', PCA(n_components=2)),
                 ('kmeans', KMeans(n_clusters=3, init='k-means++',random_state=69))])
pipe.fit(df)
result = pipe.predict(df.iloc[-1].values.reshape(1,-1))

st.subheader("Conclusions")
st.markdown("""
    **1.** The following model is used to cluster the unnsupervised data.
            
    **2.** The optimal number of clusters is **3**.
    
    **3.** These clusters are {0:"Moist", 1:"Dry", 2:"Healthy"}

    """)
if st.button("View Suggestions"):
    if result == 1:
        st.markdown("""#### The plant is dry. Please water the plant.""")
    elif result == 0:
        st.markdown("""#### The plant is moist. No need to water the plant.""")
    else:
        st.markdown("""#### The plant is healthy. No need to water the plant.""")
