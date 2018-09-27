import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df2 = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                    'sex': [1, 2, 2],
                    'level': [3, 1, 2]})

id_data = df2.values[:, :1]
transform_data = df2.values[:, 1:]

enc = OneHotEncoder()
df2_new = enc.fit_transform(transform_data).toarray()

#zu he
df2_all = pd.concat((pd.DataFrame(id_data),pd.DataFrame(df2_new)),axis=1)
print(df2_all)