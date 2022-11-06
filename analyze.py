import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

path = "./Study_Authentication/result/res_P 10_WSIZE 180.csv"

df = pd.read_csv(path)
df.columns = ["data", "predict", "actual", "NONE"]
df = df[["predict", "actual"]]
df = df.dropna()

print(len(df))

df.to_excel('./temp.xlsx', sheet_name='sheet1')

label = ['anger', 'happiness', 'fear', 'sadness', 'neutral']  # 라벨 설정
plot = plot_confusion_matrix(clf,  # 분류 모델
                             X_test_scaled, y_test,  # 예측 데이터와 예측값의 정답(y_true)
                             display_labels=label,  # 표에 표시할 labels
                             # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)
                             cmap=plt.cm.Blue,
                             normalize=None)  # 'true', 'pred', 'all' 중에서 지정 가능. default=None
plot.ax_.set_title('Confusion Matrix')
