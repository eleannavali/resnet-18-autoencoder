# Parameter configurations
BATCH_SIZE = 128
EPOCHS = 20 #200
MODEL_FILENAME = './data/model.ckpt'
EARLY_STOP_THRESH = 3 # default 30
# Learning rate for both encoder and decoder
LR= 0.001
# Load data in parallel by choosing the best num of workers for your system
WORKERS = 8