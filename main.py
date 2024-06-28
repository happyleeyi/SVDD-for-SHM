import torch #파이토치 기본모듈
from Data import get_data
from Variables import path_damaged, path_undamaged, BATCH_SIZE, rep_dim, lr_pretrain, epochs_pretrain, weight_decay_pretrain, C, S, num_class, data_saved, pretrained
from Trainer import train_model
from test import test_model


if torch.cuda.is_available():
    device = torch.device('cuda') #GPU이용

else:
    device = torch.device('cpu') #GPU이용안되면 CPU이용

print('Using PyTorch version:', torch.__version__, ' Device:', device)

Data = get_data(path_undamaged, path_damaged)
train_loader, test_loader = Data.load_data(BATCH_SIZE, data_saved)

for s in S:
    SVDD_trainer = train_model(lr_pretrain, weight_decay_pretrain, epochs_pretrain, device, train_loader, rep_dim, num_class, pretrained)
    model, net = SVDD_trainer.train(C, s)

    SVDD_tester = test_model(model, net, test_loader, num_class, device)
    SVDD_tester.confusion_mat(rep_dim, C, s)



