import numpy as np
import torch.nn as nn #신경망 모델 설계 시 필요한 함수
import torch.nn.functional as F # 자주 이용되는 함수'F'로 설정
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel

class Autoencoder(nn.Module):
    def __init__(self, rep_dim):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 1, padding = 1)
        self.dropout = nn.Dropout(0.5)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 4, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(4, 16, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256 * 64 * 1, self.rep_dim, bias=False)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, 256 * 64 * 1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        self.bn7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        self.bn8 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        self.bn9 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=2)
        self.bn10 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(16, 4, 5, bias=False, padding=2)
        self.bn11 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(4, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool2(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool2(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), 256, 64, 1)
        x = F.interpolate(F.leaky_relu(x), scale_factor=1)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn7(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn8(x)), scale_factor=1)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn9(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn10(x)), scale_factor=1)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn11(x)), scale_factor=2)
        x = self.deconv6(x)
        return x
    
class Kernel(nn.Module): # nn.Module은 모든 neural network의 base class라고 한다.
    def __init__(self, rep_dim):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 1, padding = 1)
        self.dropout = nn.Dropout(0.5)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 4, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(4, 16, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256 * 64 * 1, self.rep_dim, bias=False)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool2(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool2(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
class svdd:
    def __init__(self, C, s):
        self.C = C
        self.s = s
        self.alphas = None
        self.R2 = None
        self.Q = None
        self.X = None

    
    def train(self, X):
        C = self.C
        s = self.s
        Q = rbf_kernel(X, X, gamma=s)
        n = X.shape[0]
        P = Q + Q.T
        P = P.astype(np.double)
        q = np.diagonal(Q)
        q = q.astype(np.double)
        P = matrix(P)
        q = matrix(q.reshape(-1, 1))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), np.ones(n)*C]))
        A = matrix(np.ones((1, n)))
        b = matrix(np.ones(1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        rho = 0
        alphas = alphas.flatten()
        S = ((alphas>1e-3)&(alphas<C))
        S_idx = np.where(S)[0]
        
        R2 = alphas.dot(Q.dot(alphas))        
        if len(S_idx)>0:
            x = X[S_idx]
            K = rbf_kernel(x, X, gamma=s)
            K_ = rbf_kernel(x, x, gamma=s)
            tmp_1 = -2*np.sum(np.multiply(alphas, K)) 
            tmp_2 = np.sum(np.diag(K_))

            R2 += (tmp_1 + tmp_2)/len(S_idx)

        self.R2 = R2
        self.alphas = alphas
        self.X = X
        self.Q = Q
        return self
    
    def get_dist(self, x):
        s = self.s
        K = rbf_kernel(x, self.X, gamma=s)
        K_ = rbf_kernel(x, x, gamma=s)
        tmp_ = -2*np.sum(np.multiply(self.alphas, K), axis=1, keepdims=True) 
        offset = np.sum(np.multiply(np.matmul(np.mat(self.alphas).T, np.mat(self.alphas)), self.Q)) 

        distance = np.mat(np.diag(K_)).T+offset+tmp_
        return distance

    def predict(self, x):
        dist = self.get_dist(x)
        R2 = self.R2
        return R2-dist, dist
    