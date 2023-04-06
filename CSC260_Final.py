# Python
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy import stats


try:
    df = pd.read_csv('pmn_data.csv')
except:
    print("well that didn't work...")

print(df['spec_name'].mode()) # Coscinodiscus morphotype is the modal phytoplankton with 1482 samples
x = df.loc[(df['sampl_site'] == 'TX - Drum Bay') &  (df['spec_name'] == 'Coscinodiscus morphotype')]
y = df.loc[(df['sampl_site'] == 'TX - Port O\'Connor Fishing Pier') & (df['spec_name'] == 'Coscinodiscus morphotype')]
z = df.loc[(df['sampl_site'] == 'TX - Jims Pier') & (df['spec_name'] == 'Coscinodiscus morphotype')]
xx = df.loc[(df['sampl_site'] == 'TX - Bastrop Bay') & (df['spec_name'] == 'Coscinodiscus morphotype')]

a = df.loc[(df['sampl_site'] == 'TX - Drum Bay') &  (df['spec_name'] == 'Pseudo-nitzschia spp.')]
a = a[(np.abs(stats.zscore(df['y'])) < 3)]
b = df.loc[(df['sampl_site'] == 'TX - Port O\'Connor Fishing Pier') & (df['spec_name'] == 'Pseudo-nitzschia spp.')]
c = df.loc[(df['sampl_site'] == 'TX - Jims Pier') & (df['spec_name'] == 'Pseudo-nitzschia spp.')]


frames = [x,y,z]
framesb = [a,b,c]

Texas = pd.concat(frames)
Texas_b = pd.concat(framesb)
print(Texas)

m = Prophet()
m.fit(Texas_b)

future = m.make_future_dataframe(periods=1)
future.tail()

# # # Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecast)

# # # Python
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
