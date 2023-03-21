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

# exponential weight
def exponential_weight(length, alpha):
    exp_weight = []
    for i in range(1, length+1):
        w = alpha * (1-alpha)**i
        exp_weight.append(w)
    exp_weight = sorted(exp_weight, reverse=False)
    return(exp_weight)

# Exponentially Weighted Linear Regression
t = 900
w = exponential_weight(length=t, alpha=0.05)

x = np.arange(t)
x = x.reshape(t,1) #X는 시간에 대한 정보 reshape- 시간에 대해 한줄로 정렬
y = ts_degScore[0:t].reshape(t,1) #y는 디그레이션 펙터 정보 디그레이션 한줄로 정렬

EWLR = LinearRegression(fit_intercept=True)#fit_intercept=True 절편추가
EWLR.fit(x,y, sample_weight = w)#선형회귀 만들기 위한 x,y,w 훈련

coef = EWLR.coef_ #기울기
intercept = EWLR.intercept_#y에 대한 절편

# predefined failure threshold
failure_threshold = max(ts_degScore)

# Visualization of RUL Prediction
cl = bootlimit(tr_degScore, alpha=0.05, bootstrap=100)

fig=plt.figure(figsize=(12,4))
plt.plot(ts_degScore, color='blue')

plt.xlim([0, 1200])
plt.ylim([min(ts_degScore), max(ts_degScore)*1.005])

x = np.arange(1400)
linear_curve = coef*x + intercept
plt.plot(linear_curve[0], color='darkcyan')

plt.axhline(y=failure_threshold, color='red')
plt.axhline(y=cl, color='red')
plt.show()

predicted_failureTime= int((failure_threshold-intercept) / coef) # coef가 a라고 보면 됨필기한거있지
RUL = predicted_failureTime-t

st.markdown('예측 잔여시점: %2.2f시점' % RUL)
st.markdown('예측 잔여수명: %2.4f일' % (RUL*10/60/24))
st.markdown('실제 잔여시점: %2.2f시점' % (984-t))
st.markdown('실제 잔여수명: %2.4f일' % ((984-t)*10/60/24))

st.pyplot(fig)