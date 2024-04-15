# performance metrics

import torch
from ops.models import TSN
from ops.transforms import * 
from ptflops import get_model_complexity_info

this_weights='checkpoint/TSM_ucfcrime_RGB_mobilenetv2_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar'
#this_weights='checkpoint/TSM_ucfcrime_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar'

this_arch = 'mobilenetv2'
def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None
        
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
print(is_shift, shift_div, shift_place)


with torch.cuda.device(0):
    net = TSN(2, 1, 'RGB',
              base_model=this_arch,
              consensus_type='avg',
              img_feature_dim='225',
              #pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )
    macs, params = get_model_complexity_info(net, (1,3, 224, 224), as_strings=True,print_per_layer_stat=False, verbose=False)
    print("Using ptflops")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    

from thop import profile
model = net = TSN(2, 1, 'RGB',
              base_model=this_arch,
              consensus_type='avg',
              img_feature_dim='225',
              #pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))

from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# Draw model graph.
from torch.utils.tensorboard import SummaryWriter
    
tb = SummaryWriter()
tb.add_graph(model,input)
tb.close()









