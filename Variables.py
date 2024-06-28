path_undamaged = ["E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam1/udam1/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam2/udam2/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam3/udam3/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam4/udam4/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam5/udam5/"]
path_damaged = ["E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L1C_A/Dam1/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L1C_B/Dam4/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L3A/Dam2/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L13/Dam3/"]     # path where the data files saved

C = [0.1,0.1,0.1]
S = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]
BATCH_SIZE = 150
rep_dim = 4
lr_pretrain = 0.001
weight_decay_pretrain = 1e-6
epochs_pretrain = 200
num_class = 3
data_saved = True
pretrained = True