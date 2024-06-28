import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class test_model:
    def __init__(self, model, net, test_loader, num_class, device):
        self.model = model
        self.net = net
        self.test_loader = test_loader
        self.num_class = num_class
        self.device = device

    def result(self):
        self.net.eval()

        predict_test = []
        truevalue_test = []


        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x = x[:10]
                y = y[:10]

                for i in range(y.shape[0]):
                    truevalue_test.append(y.clone().data.cpu().numpy().tolist()[i])

                dist = []
                check = []
                for i in range(self.num_class):
                    inputs = self.net(x[:,:,:,(2-i)*8:(3-i)*8]).clone().data.cpu().numpy()                    
                    a, b = self.model[i].predict(inputs)
                    dist.append(np.array(b))
                    check.append(np.array(a))
                dist = np.squeeze(np.array(dist))
                check = np.squeeze(np.array(check))

                for i in range(x.shape[0]):
                    d = []
                    c = []
                    for j in range(self.num_class):
                        d.append(dist[j][i])
                        c.append(check[j][i])
                    check_ = np.array([np.sign(x) for x in c])
                    if sum(check_) == self.num_class:
                        predict_test.append(1)
                    else:
                        predict_test.append(np.argmin(c)+2)
                print(len(predict_test)," done")
                

        predict_test = np.array(predict_test)
        truevalue_test = np.array(truevalue_test)
        return predict_test, truevalue_test
    
    def confusion_mat(self, rep_dim, C, s):
        predict_test, truevalue_test = self.result()
        #class_names = ['normal', '1f dam', '2f dam', '3f dam']
        matrix1 = confusion_matrix(truevalue_test, predict_test)

        dataframe1 = pd.DataFrame(matrix1)


        sns.heatmap(dataframe1, annot=True, cbar=None, cmap="Blues")
        plt.title("Confusion Matrix_test"), plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        plt.savefig('confusion matrix'+' rep '+str(rep_dim)+' C '+str(C)+' s '+str(s)+'.png')
        plt.clf()

        accuracy = np.sum(predict_test==truevalue_test)/len(predict_test)
        print(" rep "+str(rep_dim)+" C "+str(C)+" s "+str(s)+"accuracy : ", accuracy)

        f = open("record.txt", "a+")
        f.write("rep dim "+str(rep_dim)+" C "+str(C)+" s "+str(s)+" accuracy : %f\n" %accuracy)
        f.close()