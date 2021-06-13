from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models
from eval import test_pretrained
from torchvision import models
import argparse

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/image_to_imagenet_c_r.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='/research/diva2/donhk/imagenet/ILSVRC2012_train/',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./data_loader/filelist/imagenet_c_and_r_filelist.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='imagenet_pretrained',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=False, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')


args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)

if args.gpu_devices == None:
    gpu_devices = '0'
else:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs, name='Imagenet_pretrained')

# G, C1, C2, opt_g, opt_c, \
# param_lr_g, param_lr_c = get_models(inputs)

target_loader.dataset.labels[target_loader.dataset.labels>1000] = 1000
test_loader.dataset.labels[test_loader.dataset.labels>1000] = 1000

model_ft = models.resnet50(pretrained=True).cuda()
ndata = target_folder.__len__()

acc_o, h_score = test_pretrained(0, test_loader, logname, n_share, model_ft,
                                 open=open, entropy=True, thr=conf.train.thr)

