import pandas as pd
data = {
    'year':[2016,2017,2018],
    'GDP rate':[2.8,3.1,3.0],
    'GDP' : ['1.637M','1.73M','1.83M']
}

df = pd.DataFrame(data, index=data['year']) # index을 추가가능

df

# row label
df.index
df.columns
df.head()

csv_data_df = pd.read_csv('E:\Git\project-python\데이터 다루기\housing.csv')
csv_data_df.head()
csv_data_df.info()

csv_data_df[['longitude','latitude']]
csv_data_df.loc[:3, ['longitude','latitude']]
csv_data_df.iloc[:3,:2]


housing["ocean_proximity"].value_counts()




# 요약
csv_data_df['longitude'].sum()
csv_data_df.describe()
# 빈도
x = pd.crosstab(index=csv_data_df.longitude, columns="count",margins=True)
print(x)


type(x)
x

import numpy as np
# plot
df = pd.DataFrame({
    
    'unif':np.random.uniform(-3,3,20),
    'norm':np.random.normal(0,1,20)
})

df.head()

df.boxplot(column=['unif','norm'])

df.index = pd.date_range('2000', freq='Y', periods=df.shape[0])
df.plot()


df.plot.scatter(x='unif', y='norm')
