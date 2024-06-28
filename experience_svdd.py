import numpy as np #선형대수 관련 함수 이용 가능 모듈
from model import svdd
from sklearn.datasets import make_classification

X_train, _ = make_classification(n_samples=1000, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0, 1],
                           class_sep=0.5, random_state=0)
X_test, Y_test = make_classification(n_samples=1000, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.5, 0.5],
                           class_sep=0.5, random_state=0)
S = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 10, 25]
for s in S:
    svdd_ = svdd(50, s).train(X_train)
    labels, _ = (np.sign(np.squeeze(svdd_.predict(X_test)))+1)/2
    acc = np.sum(labels==Y_test)/len(labels)
    print("when s value is "+str(s)+":", acc)