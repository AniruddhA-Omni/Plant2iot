import firebase_admin
from firebase_admin import credentials, firestore, db
import pandas as pd
import streamlit as st
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

##################################### Converting to dataframe and adding condition column ############################
def condition(x):
    if x["Moisture"] > 800:
        if x["Temp"] >= 33:
            return "Unbearable"
        return "Dry"
    elif x["Moisture"] < 400:
        return "Moist"
    else:
        return "Healthy"


################################################ Web App ###########################################################
st.set_page_config(page_title='Analytics App', layout='centered', initial_sidebar_state='auto', page_icon="🌱")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Plant2IoT Analytics App")
st.write("This is a web app for analytics of Plant2IoT project. The data is fetched from the Firebase Realtime Database. The data is then converted to a dataframe and then the data is visualized using Seaborn and Matplotlib.")
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
df["Condition"] = df.apply(condition, axis=1)
dx = pd.DataFrame()
dx = df

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
    return str(dx[x].iloc[-1] - dx[x].iloc[-2])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="🌡️ Temperature (°C)",
    value=str(dx["Temp"].iloc[-1]),
    delta=diff("Temp"),
)

kpi2.metric(
    label="💧 Soil Moisture",
    value= str(dx["Moisture"].iloc[-1]),
    delta=diff("Moisture"),
)

kpi3.metric(
    label="🌤️ Light Intensity",
    value=str(dx["Light"].iloc[-1]),
    delta=diff("Light"),
)

kpi4.metric(
    label="🌫️ Humidity (%)",
    value=str(dx["Hum"].iloc[-1]),
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
    df["Condition"] = df.apply(condition, axis=1)
    dx = df
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
    ax.plot(dx.index.values , dx["Moisture"],".-g",ms = 7, mfc="k")
    plt.xlabel("Time")
    plt.ylabel("Moisture")
    st.pyplot(fig)
    
    st.markdown("""**2.** The following plot shows the **Temperature** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(dx.index.values , dx["Temp"],".-r",ms = 7, mfc="g")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    st.pyplot(fig)
    
    st.markdown("""**3.** The following plot shows the **Humidity** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(dx.index.values , dx["Hum"],".-b",ms = 7, mfc="r")
    plt.xlabel("Time")
    plt.ylabel("Humidity")
    st.pyplot(fig)
    
    st.markdown("""**4.** The following plot shows the **Light** values of the plant.""")
    fig, ax = plt.subplots()
    ax.plot(dx.index.values , dx["Light"],".-k",ms = 7, mfc="gray")
    plt.xlabel("Time")
    plt.ylabel("Light Intensity")
    st.pyplot(fig)
    
    st.markdown("""**5.** The Correlation Matrix of the data is shown below.""")
    corr = dx.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.write(fig)
    
if st.button("Plot Analytics"):
    plot_analytics()
    st.markdown("""**6.** The latest data is shown below.""")
    st.write(dx.tail())
    st.write("Last Condition:",dx["Condition"].iloc[-1])
    
########################################################## Conclusion ################################################
st.subheader("Conclusions")
st.markdown("""
**1.** The plant is **healthy** when the moisture value is between 400 and 800.

**2.** The plant is **dry** when the moisture value is greater than 800.

**3.** The plant is **moist** when the moisture value is less than 400.

**4.** The plant is **unbearable** when the moisture value is greater than 800 and the temperature value is greater than or equal to 33.

""")
if st.button("View Suggestions"):
    if dx["Condition"].iloc[-1] == "Unbearable":
        st.write("The plant is unbearable. Please water the plant.")
    elif dx["Condition"].iloc[-1] == "Dry":
        st.write("The plant is dry. Please water the plant.")
    elif dx["Condition"].iloc[-1] == "Moist":
        st.write("The plant is moist. Please water the plant.")
    else:
        st.write("The plant is healthy. No need to water the plant.")
