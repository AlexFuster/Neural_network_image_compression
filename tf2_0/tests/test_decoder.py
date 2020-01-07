import sys
sys.path.append('../src')
from decoder import Decoder

if __name__=="__main__":
    dataset_path=sys.argv[1]
    if len(sys.argv)==3:
        checkpoint_path=sys.argv[2]
    else:
        checkpoint_path='../checkpoints/decoder'

    decoder=Decoder()
    decoder.compress(dataset_path,checkpoint_name)