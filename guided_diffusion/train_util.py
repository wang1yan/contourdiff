import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader = dataloader
       # self.discriminiator = discriminiator
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt_model = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
       # self.opt_discriminiator = AdamW(
       #     self.discriminiator.parameters(), lr=self.lr, weight_decay=self.weight_decay
       # )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            #self.discriminiator = discriminiator.to(dist_util.dev())
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch, cond, name = next(data_iter)
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch, cond, name = next(data_iter)

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        img = batch
        label = cond
        batch=th.cat((batch, cond), dim=1) # 将原始图和标签图拼接
        cond={}
        #dis_sample = self.dis_forward_backward(batch, cond, img.to(dist_util.dev()), label.to(dist_util.dev())) 
        gen_sample = self.gen_forward_backward(batch, cond, img.to(dist_util.dev()), label.to(dist_util.dev()))
        took_step = self.mp_trainer.optimize(self.opt_model) # 优化
       # self.opt_discriminiator.step()
        if took_step: # 更新ema
            self._update_ema()
        self._anneal_lr() # 调整学习率
        self.log_step() # 记录日志
        return gen_sample

    def dis_forward_backward(self, batch, cond, img, label):
        sigmoid = nn.Sigmoid()
        #训练判别器
        self.opt_discriminiator.zero_grad() # 梯度清零
        real_output = self.discriminiator(img).to(dist_util.dev())
        real_label = self.discriminiator(label).to(dist_util.dev())
        #真实图片的损失
        loss_dis = nn.functional.binary_cross_entropy_with_logits(sigmoid(real_output), real_label)
        real_score = sigmoid(real_output)
        #生成图片的损失
        dis_sample = self.diffusion.p_sample_loop(
            self.ddp_model,
            batch.shape,
            model_kwargs=cond,
            clip_denoised=True,
            device = dist_util.dev(),
            progress=False,
        )
        fake_output = self.discriminiator(dis_sample).to(dist_util.dev())
        loss_gen = nn.functional.binary_cross_entropy_with_logits(sigmoid(fake_output), real_label)
        fake_score = sigmoid(fake_output)
        logger.logkv("fake_score", fake_score.mean().item())
        logger.logkv("real_score", real_score.mean().item())
        # 计算判别器的损失
        loss_D = loss_gen + loss_dis
        logger.logkv("loss_D", loss_D.item())
        loss_D.backward()
        self.opt_discriminiator.step()
        return dis_sample
        
    def gen_forward_backward(self, batch, cond , img, label):
        #训练生成器
        self.opt_model.zero_grad() # 梯度清零
        # 将该批次划分为更小的批次
       
        # 对时间步和权重进行采样
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev()) #均匀分布
        # 传入一部分的参数到损失函数中，创建为新的函数compute_losses
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            self.classifier,
            batch,
            t,
            model_kwargs=cond,)
        # 计算损失
        
        with self.ddp_model.no_sync():
            losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1[0]["loss"].detach()
            )# 更新采样分布
        losses = losses1[0]
        gen_sample = losses1[1]
        #output = self.discriminiator(gen_sample).to(dist_util.dev())
        #real_label = self.discriminiator(label).to(dist_util.dev())
        #loss_dis = th.nn.functional.binary_cross_entropy_with_logits(output, real_label)
        loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()
        #loss_G = (loss_dis * 10 + (losses["loss"] * weights)).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
        self.mp_trainer.backward(loss)
        for name, param in self.ddp_model.named_parameters():
            if param.grad is None:
                print(name)
        return gen_sample
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt_model.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt_model.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
