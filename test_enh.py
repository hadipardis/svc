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
from dcvc.models.video_net_dmc import DMC, DMC_Original, DMC_Turbo, DMC_STE
from dcvc.models.priors import IntraNoAR
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
        for idx in (range(0,no_frames)):  #tqdm
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
def load_base_model_video(yolo_model, weight_path="./weights/checkpoint_best_loss.pth.tar"):
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
    checkpoint_net = torch.load(weight_path, map_location=(lambda storage, loc: storage))
    net.load_state_dict(checkpoint_net['state_dict'], strict=True)

    return net
#=======================================================================================================================
@torch.no_grad()
def encode_the_enhancement_layer(rd_point, root_path, out_path, prefix, tot_frames, GOP):
    device = select_device('', batch_size=1)
    yolo_model = DetectMultiBackend(weights=ROOT/'yolov5s.pt', device=device, dnn=False, data=ROOT/'data/hevc_class_b_coded.yaml', fp16=False)
    #load the base layer's model
    base_model_checkpoint = "./checkpoints/base/checkpoint_base_1.pth.tar"
    net_base_video = load_base_model_video(yolo_model, base_model_checkpoint)
    net_base_video = net_base_video.module
    net_base_video = net_base_video.cuda()
    net_base_video.eval()

    #create the enhancement layer's model
    net = DMC_Turbo()

    if rd_point==1:
        checkpoint = torch.load("./checkpoints/enh/checkpoint_enh_1.pth.tar",map_location=(lambda storage, loc: storage))
    if rd_point==2:
        checkpoint = torch.load("./checkpoints/enh/checkpoint_enh_2.pth.tar",map_location=(lambda storage, loc: storage))
    if rd_point==3:
        checkpoint = torch.load("./checkpoints/enh/checkpoint_enh_3.pth.tar",map_location=(lambda storage, loc: storage))
    if rd_point==4:
        checkpoint = torch.load("./checkpoints/enh/checkpoint_enh_4.pth.tar",map_location=(lambda storage, loc: storage))

    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net = net.cuda()
    net.eval()
    net.training = False
    align = Alignment().cuda()

    all_SSIM = AverageMeter()
    all_RATE = AverageMeter()
    all_MSE = AverageMeter()
    ms_ssim = MS_SSIM(reduction='mean', data_range=1.).to('cuda')

    SSIM = AverageMeter()
    RATE = AverageMeter()
    MSE = AverageMeter()

    print("Generating the enhancement frames...")
    for i in tqdm(range(0,tot_frames)):
        p = '%s%s%03d.png' %(root_path,prefix,i)
        frame = Image.open(p).convert("RGB")
        orig_x = transforms.ToTensor()(frame).unsqueeze(0).cuda()
        im_shape = orig_x.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        x = align.align(orig_x)

        if i % GOP==0:
            rec_frame_base, likelihoods_base, _ = net_base_video.if_model(x)
            rec_frame_base = rec_frame_base.clamp(0, 1)
            rate_base = estimate_bpp(likelihoods_base, input=orig_x).mean().item()

            RATE.update(rate_base)
            mse = torch.mean((align.resume(rec_frame_base)-orig_x)**2)
            MSE.update(mse)
            eval_msssim = ms_ssim(align.resume(rec_frame_base), orig_x).mean().item()
            SSIM.update(eval_msssim)

            img = torch2img(align.resume(rec_frame_base).detach().cpu())
            img.save('%s/%s%03d.png' % (out_path, prefix, i))

            out_frame = rec_frame_base
            ref_base = rec_frame_base
            frame_index = 1

            ref_feature = None
            dpb = {"y_hat":None}
            ref_frame = out_frame
        else:
            if frame_index==1:
                net_base_video.frame_buffer = [ref_base]

            #code P frame - the base layer
            rec_frame_base, likelihoods_base = net_base_video.forward_pair(ref_base, x, frame_index)
            img = torch2img(align.resume(rec_frame_base).detach().cpu())

            rate_base = estimate_bpp(likelihoods_base, input=orig_x).mean().item()
            rec_frame_base = rec_frame_base.clamp(0,1)
            ref_base = rec_frame_base
            net_base_video.frame_buffer.append(ref_base)

            # code P frame - the enh layer
            result = net.forward(x, ref_frame, ref_feature, rec_frame_base, dpb)
            dpb = result["dpb"]

            ref_frame = result['recon_image']
            out_frame = ref_frame
            ref_feature = result['feature']

            frame_index = frame_index + 1

            bpp_y = result["bit_y"] / pixel_num
            bpp_z = result["bit_z"] / pixel_num
            bpp_mv_y = result["bit_mv_y"] / pixel_num
            bpp_mv_z = result["bit_mv_z"] / pixel_num
            bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
            # bpp = result["bpp"]


            RATE.update(bpp + rate_base)
            mse = torch.mean((align.resume(result['recon_image'])-orig_x)**2)
            # mse = torch.mean((result['recon_image'] - x) ** 2)
            MSE.update(mse)
            eval_msssim = ms_ssim(align.resume(result['recon_image']), orig_x).mean().item()
            # eval_msssim = ms_ssim(result['recon_image'], x).mean().item()
            SSIM.update(eval_msssim)

            img = torch2img(align.resume(result['recon_image']).detach().cpu())
            img.save('%s/%s%03d.png' % (out_path, prefix, i))


        # Update frame buffer
        if len(net_base_video.frame_buffer) == 4:
            net_base_video.frame_buffer.pop(0)
            assert len(net_base_video.frame_buffer) == 3, str(len(net_base_video.frame_buffer))


    mean_mse = MSE.avg
    mean_rate = RATE.avg
    psnr = 10*log10(1/mean_mse)
    mean_ssim = SSIM.avg
    print("PSNR: %f, BPP:%f, SSIM: %f" % (psnr, mean_rate, mean_ssim))
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
    encode_the_enhancement_layer(args.checkpoint_number, args.inp_path, args.out_path, args.prefix, args.no_frames, args.gop)  # main function for testing the enh layer
