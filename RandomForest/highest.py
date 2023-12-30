from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

movie = pd.read_csv("../data/data.csv")
# split the data into features and results
X = movie[movie.columns[3:23]]
y = movie[movie.columns[-1]]

# spliting the data into Train Test and Validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, shuffle=True)
model = RandomForestRegressor(n_estimators=100, max_depth=5,
                              min_samples_split=30, min_samples_leaf=3,
                              random_state=22)
model.fit(X_train, y_train)
test_mae = mean_absolute_error(model.predict(X_test), y_test)
print(f'regressor: train mae:{mean_absolute_error(model.predict(X_train), y_train)}, test mae:{test_mae}')


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
movie['rating'] = pd.qcut(movie['rating'], q=5, labels=False, duplicates='drop')
X = movie[movie.columns[3:23]]
y = movie[movie.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle=True)
model = RandomForestClassifier(n_estimators=140,max_depth=10,min_samples_split=16,min_samples_leaf=4,random_state=44)
model.fit(X_train, y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
print(f'classifier: test_acc : {test_acc:.2f}')