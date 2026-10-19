import pandas as pd
import numpy as np

d1={'Social_media_followers':[1000000, np.nan, 2000000, 1310000, 1700000, np.nan, 4100000, 1600000, 2200000, 1000000],
    'Sold_out':[1,0,0,1,0,0,0,1,0,1]}
df1 = pd.DataFrame(data=d1)
df1 

X1 = df1[['Social_media_followers']]
y1 = df1[['sold_out']]

from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=11)
X1_train.shape
X1_test.shape

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

from sklearn.pipeline import make_pipeline
pip = make_pipeline(imputer, lr)
pip.fit(X1_test,X1_train)
pip.score(X1_train, y1_train)
pip.score(X1_test, y1_test)
pip.named_steps.simpleimputer.statistics_
pip.named_steps.logisticsregression.coef_