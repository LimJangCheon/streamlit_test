import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_icon="💻",
    page_title="스트림릿에서 시각화!",
    layout="wide",
)

st.subheader("2nd_test 시각화")

rms_df = pd.read_csv(r'C:\Users\dlawk\RUL\NASA_Bearing\2nd_test\\RMS_bearing.csv')

fig=plt.figure(figsize=(12,4))
plt.plot(rms_df.values)
plt.show()

st.pyplot(fig)