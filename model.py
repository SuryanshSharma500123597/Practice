from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

iris = load_iris()

model = RandomForestClassifier()
model.fit(iris.data, iris.target)

joblib.dump(model, "model.pkl")