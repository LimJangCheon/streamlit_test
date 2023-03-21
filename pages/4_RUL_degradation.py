import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_icon="💻",
    page_title="스트림릿에서 시각화2!",
    layout="wide",
)

st.subheader("2nd_test Degradation")

rms_df = pd.read_csv(r'C:\Users\dlawk\RUL\NASA_Bearing\2nd_test\\RMS_bearing.csv')

trdat = rms_df.values[0:400,:]
tsdat = rms_df.values

trScore_arr = np.zeros([trdat.shape[0], trdat.shape[1]])
tsScore_arr = np.zeros([tsdat.shape[0], trdat.shape[1]])

lr = LinearRegression()

input_idx = np.arange(trdat.shape[1]).tolist()
for idx in input_idx:
    input_idx = np.arange(trdat.shape[1]).tolist()
    input_idx.remove(idx)
    
    lr.fit(trdat[:,input_idx], trdat[:,idx])
    
    # 각 변수 별 train/test score
    trScore = lr.predict(trdat[:,input_idx])
    tsScore = lr.predict(tsdat[:,input_idx])
    
    trScore_arr[:,idx] = trScore
    tsScore_arr[:,idx] = tsScore

# 각 변수 별 통합 train/test score
integrated_trScore = np.sqrt(np.sum(trScore_arr**2, axis=1))
integrated_tsScore = np.sqrt(np.sum(tsScore_arr**2, axis=1))

def bootlimit(stat, bootstrap, alpha):    
    alpha = 100 - alpha*100
    samsize = len(stat)
    sampled_limit = []
    for i in range(bootstrap):
        sampled_limit.append(np.percentile(np.random.choice(stat, samsize, replace=True), alpha))
    limit = np.mean(sampled_limit)
    return(limit)

cl = bootlimit(integrated_trScore, bootstrap=100, alpha=0.05)

outidx = np.where(integrated_tsScore>cl)[0]

plt.figure(figsize=(12,4))
plt.plot(integrated_tsScore, color='blue')
# plt.plot(outidx, np.repeat(max(integrated_tsScore)*1.1, len(outidx)), 'x', color='red', alpha=0.5)
plt.axhline(y=cl, color='red')

# for idx in outidx:
  #   plt.axvline(x=idx, color='red', linestyle='-', alpha=0.1)



# 이상감지 잔차 값 넣은 컨트롤리미트
plt.show()

#고장날때쯤생기는 진동(큰분산) 완화를 위해 스트레스 누적시킴(ppt참고) l2놈 적용결과 (l2놈=mset)
l2norm_trScore = np.sqrt(np.sum(trScore_arr**2, axis=1))
tr_degScore = np.cumsum(l2norm_trScore) / np.arange(1,401,1)

l2norm_tsScore = np.sqrt(np.sum(tsScore_arr**2, axis=1))
ts_degScore = np.cumsum(l2norm_tsScore) / np.arange(1,985,1)

cl = bootlimit(tr_degScore, alpha=0.05, bootstrap=100)
#디그라데이션결과를 넣어 만든 컨트롤리밑
fig=plt.figure(figsize=(12,4))
plt.plot(ts_degScore, color='blue')
plt.axhline(y=cl, color='red')
plt.show()


st.pyplot(fig)