import os
from tqdm import tqdm
from pathlib import Path
import argparse
import torch
import yaml
import copy
import time
from PIL import Image
from collections import defaultdict
from typing import Dict, Union, List
from loguru import logger
from omegaconf import OmegaConf
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter
from src.utils import initialize_from_config, set_seed, count_parameters, get_obj_from_str, make_dirs, cal_elasped_time


class Trainer:
    def __init__(self, device, config_path, overfit_batch=False, ckpt_path=None, pretrained_path=None):
        config = OmegaConf.load(config_path)
        
        self.config = config
        self.config_path = config_path
        
        dataset = initialize_from_config(self.config.dataset)
        self.dataset=dataset
        if hasattr(self.config.training, 'sampler'):
            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=get_obj_from_str(self.config.training.sampler.target)(
                    dataset,
                    shuffle=True,
                    batch_size=self.config.training.batch_size,
                    drop_last=True
                ),
                collate_fn=dataset.collate_function if self.config.training.use_collate_fn else None,
                pin_memory=True,
                num_workers=self.config.training.num_workers
            )
        else:
            self.loader = torch.utils.data.DataLoader(
                dataset,
                self.config.training.batch_size,
                shuffle=True,
                pin_memory=True,
                collate_fn=dataset.collate_function if self.config.training.use_collate_fn else None,
                num_workers=self.config.training.num_workers
            )
        self.batch_images = copy.deepcopy(next(iter(self.loader)))
        
        if hasattr(self.config, 'testdataset') and self.config.training.valid:
            test_dataset = initialize_from_config(self.config.testdataset)
            self.test_loader = torch.utils.data.DataLoader(
                            test_dataset,
                            self.config.training.val_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=dataset.collate_function if self.config.training.use_collate_fn else None,
                            num_workers=self.config.training.num_workers
                        )
            self.top_k_track = []
            self.max_top_k = self.config.metric.save_top_k
        else:
            self.test_loader = None

        self.config.text_encoder.params.input_size = dataset.tokenizer.vocab_size
            
        pipeline = get_obj_from_str(self.config.pipeline.target)(
            vae_config=self.config.vae.vae_pretrained_path,
            diffusion_config=self.config.diffusion,
            style_extractor_config=self.config.style_extractor,
            text_encoder_config=self.config.text_encoder,
            style_constrastive_config=self.config.constrastive,
            content_constrastive_config=self.config.content_constrastive,
            loss_balancing=self.config.training.loss_balancing,
        )
        pipeline.configure_optimizers(self.config.training.base_lr)
        pipeline = pipeline.to(device)
        
        self.cur_iter = None
        self.cur_epoch = None
        self.best_loss = float('inf')
        self.continue_training = False
        if ckpt_path is not None:
            cur_iter, _, best_loss = pipeline.load_checkpoint(ckpt_path)
            self.cur_iter = cur_iter
            if best_loss is not None:
                self.best_loss = best_loss
            self.continue_training = True
            
        if pretrained_path is not None:
            pipeline.load_state_dict(pretrained_path)
            
        self.pipeline = pipeline
        if self.config.training.update_ema:
            self.pipeline.setup_ema()
        if self.config.training.amp:
            self.pipeline.setup_amp()

        if hasattr(self.config.training, 'scheduler'):
            self.pipeline.setup_scheduler(
                self.config.training.scheduler,
                num_training_steps = self.config.training.iteration
            )
            
        self.total_iterations = self.config.training.iteration
        self.device = device
        self.overfit_batch = overfit_batch
        
    def setup_exp(self):
        timestamp = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp/1000))
        #Setup experiment log dir
        config_rel_path = Path(self.config_path).absolute().relative_to(Path(__file__).absolute().parent.joinpath('config')).parent.joinpath(Path(self.config_path).stem)
        exp_dir = Path(self.config.training.output_dir).joinpath(config_rel_path, timestamp)
        make_dirs(exp_dir)
        
        #Save current config information
        with open(exp_dir.joinpath(Path(self.config_path).name), 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)
        
        #Setup logger log path
        logger.add(exp_dir.joinpath('train.log'), format="{time} {level} {message}", level="INFO")
        
        #Setup tensorboard log path
        tb_dir = exp_dir.joinpath('tbrun')
        make_dirs(tb_dir)
        self.writer = SummaryWriter(log_dir=tb_dir)
        
        #Setup checkpoint dir
        self.ckpt_dir = exp_dir.joinpath('checkpoint')
        make_dirs(self.ckpt_dir)
        
        #Setup image visualize dir
        self.log_image = exp_dir.joinpath('images')
        make_dirs(self.log_image)
        
    def to_device(self, batch: Dict[str, Union[torch.Tensor, List[str]]]):
        return {keyword:(value.to(self.device) if isinstance(value, torch.Tensor) else value) for keyword, value in batch.items()}
            
    def train(self):
        num_params = count_parameters(self.pipeline)
        for key in num_params:
            logger.info('Number of trainable parameters of {} : {}', key, num_params[key])    
    
        print('Training started....')
        cur_iter = self.cur_iter or 0
        
        dataloader = iter(self.loader)
        
        self.pipeline.train()
        total_loss = defaultdict(list)
        
        start_time = time.time()
        while True:
            if self.continue_training and self.best_loss > 10000:
                self.pipeline.eval()
                val_loss = self.run_validation(cur_iter)
                self.continue_training = False
                time_ = cal_elasped_time(time.time(), start_time)
                logger.info('Iteration: {}-Elapsed time: {}-Val loss: {}-Best loss: {}', cur_iter, time_, val_loss, self.best_loss)
                self.pipeline.train()
                
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(self.loader)
                batch = next(dataloader)
            
            batch = self.to_device(batch)
            
            #Perform forward and backward update parameter and update learning rate if has any
            loss = self.pipeline.forward_backward_update(**batch, iteration=cur_iter)
            if loss is None:
                continue
                
            for k in loss:
                total_loss[k].append(loss[k].item())
                                    
            if self.config.training.update_ema:
                self.pipeline.step_ema()
                                  
            for k in loss:
                self.writer.add_scalar(k, loss[k].item(), cur_iter+1)
                    
            if self.overfit_batch:
                break
            
            cur_iter += 1
            
            if cur_iter % self.config.training.log_interval == 0 or cur_iter % self.config.training.save_interval == 0:
                loss_info = ''
                for idx, k in enumerate(total_loss):
                    msg = 'Train {}: {}'.format(k, sum(total_loss[k])/len(total_loss[k]))
                    if idx < len(total_loss) - 1:
                        msg += '-'
                    loss_info += msg
                    
            if cur_iter % self.config.training.log_interval == 0:    
                self.pipeline.eval()
            
                buffer_idx = int(cur_iter // self.config.training.log_interval // self.config.training.log_image_buffer)
                buffer_dir = self.log_image.joinpath(str(buffer_idx))
                make_dirs(buffer_dir)
                self.pipeline.log_images(batch=self.to_device(self.batch_images), save_path=buffer_dir.joinpath(f'iter_{cur_iter}.png'))

                time_ = cal_elasped_time(time.time(), start_time)
                logger.info('Iteration: {}-Elapsed time: {}-{}', cur_iter, time_, loss_info)
                
                self.pipeline.train()
                
            if cur_iter % self.config.training.save_interval == 0:                    
                self.pipeline.eval()
                
                self.pipeline.save_state_dict(self.ckpt_dir.joinpath(f'iter_{cur_iter}.pth'))

                if self.test_loader is not None:
                    val_loss = self.run_validation(cur_iter)
                    self.pipeline.save_checkpoint(self.ckpt_dir.joinpath('ckpt.pth'), cur_iter, self.best_loss)
                    time_ = cal_elasped_time(time.time(), start_time)
                    logger.info('Iteration: {}-Elapsed time: {}-{}-Val loss: {}-Best loss: {}', cur_iter, time_, loss_info, val_loss, self.best_loss)
                else:
                    self.pipeline.save_checkpoint(self.ckpt_dir.joinpath('ckpt.pth'), cur_iter)
                    
                self.pipeline.train()
                
            if cur_iter >= self.total_iterations:
                break
            
    @torch.inference_mode()
    def validation_step(self, dataloader):
        val_losses = 0.0
        count = 0
        dataloader = iter(dataloader)
        
        while True:
            try:
                batch = next(dataloader)
            except StopIteration:
                break
                
            batch = self.to_device(batch)
            val_loss = self.pipeline.validation_step(batch)
            val_losses += val_loss.item()
            count += 1
            
        final_loss = val_losses / count
            
        return final_loss
    
    @torch.inference_mode()
    def run_validation(self, iter_count):        
        val_loss = self.validation_step(self.test_loader)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            if len(self.top_k_track) == self.max_top_k:
                remove_ckpt = self.top_k_track.pop(0)
                for ckpt in remove_ckpt:
                    os.remove(ckpt.as_posix())
            ckpt_path = self.ckpt_dir.joinpath(f'iter_{iter_count}_loss{self.best_loss}.pth')
            if self.config.training.update_ema:
                ema_ckpt_path = self.ckpt_dir.joinpath(f'iter_{iter_count}_loss{self.best_loss}_ema.pth')
                self.top_k_track.append((ema_ckpt_path, ckpt_path))
                self.pipeline.save_state_dict(ckpt_path)
                self.pipeline.save_state_dict_ema(ema_ckpt_path)
            else:
                self.top_k_track.append((ckpt_path, ))
                self.pipeline.save_state_dict(ckpt_path)
        
        return val_loss

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='config/constant.yaml')
    parser.add_argument('--overfit1batch', action='store_true', default=False)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--pretrained_path', default=None)
    args = parser.parse_args()
    set_seed(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(device, args.config_path, args.overfit1batch, args.ckpt_path, args.pretrained_path)
    trainer.setup_exp()
    trainer.train()
