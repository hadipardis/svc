import argparse
import json
import os
import sys
from pathlib import Path
import shutil
import math
import numpy as np
import torch
from tqdm import tqdm
from math import log10
import torch
from networks_canf import __CODER_TYPES__, AugmentedNormalizedFlowHyperPriorCoder
from PIL import Image
from Utils import Alignment, BitStreamIO
import torch.nn.functional as F
from entropy_models import EntropyBottleneck, estimate_bpp
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms as transforms
import random
from torch import nn, optim
from typing import Any, Dict, List, Tuple, Union
from flownets import PWCNet, SPyNet
from SDCNet import MotionExtrapolationNet
from Models import Refinement, RefinementAttention
import yaml
import copy
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from loss import MS_SSIM, PSNR
from Model.Model import Model
from util.sampler import Resampler
#=======================================================================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#=======================================================================================================================
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#=======================================================================================================================
def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())
#=======================================================================================================================
# Creating our proposed model
class CompressModel(nn.Module):#nn.Module
    """Basic Compress Model"""

    def __init__(self):
        super(CompressModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).mean()
#=======================================================================================================================
class Pframe(CompressModel):
    def __init__(self, mo_coder, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').cuda()

        self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_QE=True,
                                                               use_affine=False,
                                                               use_context=True, condition='GaussianMixtureModel',
                                                               quant_mode='round').cuda()
        self.MENet = PWCNet(trainable=True).cuda()

        self.MWNet = MotionExtrapolationNet(sequence_length=3).cuda()
        self.MWNet.__delattr__('flownet')

        self.Motion = mo_coder.cuda()
        self.CondMotion = cond_mo_coder.cuda()

        self.Resampler = Resampler().cuda()
        self.MCNet = Refinement(6, 64, out_channels=3).cuda()

        self.Residual = res_coder.cuda()

        self.frame_buffer = list()
        self.flow_buffer = list()
        self.align = Alignment().cuda()

    def motion_forward(self, ref_frame, coding_frame, p_order=1):
        # To generate extrapolated motion for conditional motion coding or not
        # "False" for first P frame (p_order == 1)
        predict = p_order > 1
        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2

            # Update frame buffer ; motion (flow) buffer will be updated in self.MWNet
            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]

            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            pred_frame, pred_flow = self.MWNet(frame_buffer, self.flow_buffer if len(self.flow_buffer) == 2 else None,
                                               True)

            flow = self.MENet(ref_frame, coding_frame)

            # Encode motion condioning on extrapolated motion
            flow_hat, likelihood_m, _, _ = self.CondMotion(flow, xc=pred_flow, x2_back=pred_flow,
                                                           temporal_cond=pred_frame)

        # No motion extrapolation is performed for first P frame
        else:
            flow = self.MENet(ref_frame, coding_frame)
            # Encode motion unconditionally
            flow_hat, likelihood_m = self.Motion(flow)

        warped_frame = self.Resampler(ref_frame, flow_hat)
        mc_frame = self.MCNet(ref_frame, warped_frame)

        self.MWNet.append_flow(flow_hat)

        return mc_frame, likelihood_m, flow_hat

    #forward function for testing
    def forward_pair(self, ref_frame, coding_frame, p_order=1):
        mc_frame, likelihood_m, flow_hat = self.motion_forward(ref_frame, coding_frame, p_order)

        reconstructed, likelihood_r, x2, _ = self.Residual(coding_frame, xc=mc_frame, x2_back=mc_frame,
                                                            temporal_cond=mc_frame)
        reconstructed = reconstructed.clamp(0, 1)
        likelihoods = likelihood_m + likelihood_r

        return reconstructed, likelihoods

    @torch.no_grad()
    def test(self, no_frames, inp_path, prefix, out_path, gop):
        align = Alignment().cuda()
        RATE = AverageMeter()
        MSE = AverageMeter()
        SSIM = AverageMeter()
        ms_ssim = MS_SSIM(reduction='mean', data_range=1.).to('cuda')
        for idx in tqdm(range(0,no_frames)):
            filename = '%s/%s%03d.png' % (inp_path, prefix, idx)
            coding_frame = ToTensor()(Image.open(filename)).unsqueeze(0).cuda()

            if idx%gop==0: # I-frames
                rec_frame, likelihoods, _ = self.if_model(align.align(coding_frame))
                rec_frame = align.resume(rec_frame.cuda()).clamp(0, 1)
                mse = self.criterion(rec_frame, coding_frame).mean().item()
                MSE.update(mse)
                eval_msssim = ms_ssim(rec_frame, coding_frame).mean().item()
                SSIM.update(eval_msssim)
                rate = estimate_bpp(likelihoods, input=rec_frame).mean().item()
                RATE.update(rate)
                frame_idx = 1
            else: #P-frames
                if frame_idx == 1:
                    self.frame_buffer = [align.align(ref_frame)]

                rec_frame, likelihoods = self.forward_pair(align.align(ref_frame), align.align(coding_frame), frame_idx)

                rec_frame = rec_frame.clamp(0, 1)
                self.frame_buffer.append(rec_frame)

                # Back to original resolution
                rec_frame = align.resume(rec_frame)
                rate = estimate_bpp(likelihoods, input=coding_frame).mean().item()
                RATE.update(rate)

                mse = self.criterion(rec_frame, coding_frame).mean().item()
                MSE.update(mse)

                eval_msssim = ms_ssim(rec_frame, coding_frame).mean().item()
                SSIM.update(eval_msssim)

                frame_idx = frame_idx + 1

            # Update frame buffer
            if len(self.frame_buffer) == 4:
                self.frame_buffer.pop(0)
                assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))

            img = torch2img(rec_frame.detach().cpu())
            img.save('%s/%s%03d.png' % (out_path, prefix, idx))

            ref_frame = rec_frame

        m = MSE.avg
        r = RATE.avg
        s = SSIM.avg
        return m, s, r
#=======================================================================================================================
@torch.no_grad()
def encode_the_base_layer(base_checkpoint, no_frames, inp_path, prefix, out_path, gop):
    # First Load the P frame network
    motion_coder_conf = './config/DVC_motion.yml'
    cond_motion_coder_conf = './config/CANF_motion_predprior.yml'
    residual_coder_conf = './config/CANF_inter_coder.yml'

    mo_coder_cfg = yaml.safe_load(open(motion_coder_conf, 'r'))
    mo_coder_arch = __CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])

    cond_mo_coder_cfg = yaml.safe_load(open(cond_motion_coder_conf, 'r'))
    cond_mo_coder_arch = __CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])

    res_coder_cfg = yaml.safe_load(open(residual_coder_conf, 'r'))
    res_coder_arch = __CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])

    net = Pframe(mo_coder, cond_mo_coder, res_coder).cuda()
    net = torch.nn.DataParallel(net)

    device = select_device('', batch_size=1)

    yolo_model = DetectMultiBackend(weights=ROOT/'yolov5s.pt', device=device, dnn=False, data=ROOT/'data/hevc_class_b_coded.yaml', fp16=False)
    net.module.feature_extractor = copy.deepcopy(yolo_model)

    checkpoint_net = torch.load(base_checkpoint, map_location=(lambda storage, loc: storage))
    net.load_state_dict(checkpoint_net['state_dict'], strict=True)
    net = net.cuda()
    net.eval()

    print("Generating the base frames...")
    m, s, r = net.module.test(no_frames, inp_path, prefix, out_path, gop)
    psnr = 10*log10(1/m)
    print("BPP: ", r, " PSNR: ", psnr, " SSIM: ", s)

    return

#=======================================================================================================================
def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")

    parser.add_argument(
        '--inp_path',
        type=str,
        default="\\input\\",
    help = 'Path to the input directory (default: .\\input\\)'
    )

    parser.add_argument(
        '--out_path',
        type=str,
        default="\\out\\base\\",
    help = 'Path to the input directory (default: .\\out\\base\\)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
    help = 'prefix of the video frame names (e.g. Parkscene_)'
    )

    parser.add_argument(
        '--checkpoint_number',
        type=int,
        default=1,
        help='The checkpoint number (1,2,3,4) (default: 1)'
    )

    parser.add_argument(
        '--gop',
        type=int,
        default=32,
        help='GOP or intra period size (default: 32)'
    )

    parser.add_argument(
        '--no_frames',
        type=int,
        default=100,
        help='Number of frames (default: 100)'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
#=======================================================================================================================
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.checkpoint_number == 1:
        # the lowest rate
        base_checkpoint_path = "./checkpoints/base/checkpoint_base_1.pth.tar"
    if args.checkpoint_number == 2:
        base_checkpoint_path = "./checkpoints/base/checkpoint_base_2.pth.tar"
    if args.checkpoint_number == 3:
        base_checkpoint_path = "./checkpoints/base/checkpoint_base_3.pth.tar"
    if args.checkpoint_number == 4:
        # the highest rate
        base_checkpoint_path = "./checkpoints/base/checkpoint_base_4.pth.tar"

    encode_the_base_layer(base_checkpoint_path, args.no_frames, args.inp_path, args.prefix, args.out_path, args.gop)


