
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bsds_dataset_loader import BsdsDataset
from guided_diffusion.biped_dataset_loader import BipedDataset
from guided_diffusion.nyud_dataset_loader import NyudDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=45461)
import torchvision.transforms as transforms

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'BIPED':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = BipedDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BSDS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = BsdsDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = NyudDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    #discriminiator = Discriminator()
    #DGenerator = DiffGenerator(args)
    #model, diffusion = DGenerator()
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
        #discriminiator = th.nn.DataParallel(discriminiator,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        #discriminiator.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
        #discriminiator.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        #discriminiator=discriminiator,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = '',
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=2000,
        resume_checkpoint="/media/yang/Pytorch/wang/edgediff/results/NYUDresults/savedmodel507000.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = "0,1,2",
        out_dir='/root/ww/yuanedgeDiff/results'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
