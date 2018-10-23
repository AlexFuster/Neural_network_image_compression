import sys
sys.path.append('../src')
from training import training_CAE
from utils import read_dataset, get_next_log_name
import os

if len(sys.argv)==5:
    _,dataset_path,max_epochs,batch_size,_=sys.argv
    from_checkpoint=True
else:
    _,dataset_path,max_epochs,batch_size=sys.argv
    from_checkpoint=False

X,_=read_dataset(dataset_path,istrain=True)
dataset_name=dataset_path.split('/')[-1]

try:
    os.makedirs('log_dir_'+dataset_name)
except:
    pass

log_dir=get_next_log_name('log_dir_'+dataset_name+'/',from_checkpoint)

try:
    os.makedirs(log_dir)
except:
    pass



checkpoint_dir='checkpoints/'+dataset_name+'/'
try:
    os.makedirs(checkpoint_dir)
except:
    pass
last_epoch_file='last_epoch_'+dataset_name+'.txt'

training_CAE(log_dir,checkpoint_dir,last_epoch_file).train(X['train'],X['test'],int(max_epochs),int(batch_size),from_checkpoint,dataset_name)