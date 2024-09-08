# Nạp các gói thư viện cần thiết
import pandas as pd
from sklearn import svm
# Đọc dữ liệu iris từ UCI (https://archive.ics.uci.edu/ml/datasets/Iris)
# hoặc từ thư viện scikit-learn
# Tham khảo https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
columns=["Petal length","Petal Width","Sepal Length","Sepal Width"];
df = pd.DataFrame(iris.data, columns=columns)
y = iris.target
print(df.describe())
print("\n")
print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
print(df.isnull().sum())
# Sử dụng nghi thức kiểm tra hold-out
# Chia dữ liệu ngẫu nhiên thành 2 tập dữ liệu con:
# training set và test set theo tỷ lệ 70/30
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
# Xây dựng mô hình svm sử dụng hàm nhân (kernel) là RBF
# SVC là viết tắt của từ Support Vector Classification
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
prediction = model.predict(X_test)
#print(prediction)
# Tính độ chính xác
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
model.score(X_test, y_test))

#save model
import pickle
filename = 'B2106784_model.pkl'
pickle.dump(model, open(filename, 'wb'))