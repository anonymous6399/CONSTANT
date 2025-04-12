import argparse
import torch
import yaml
import time
from PIL import Image
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from src.utils import initialize_from_config, set_seed, make_dirs, make_image_grid, get_obj_from_str
import json


class Runner:
    def __init__(self, device, root_path, weight_path, batchsize, overide_params=[], save_by_name=False):
        self.device = "cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu"
        self.batchsize = batchsize
        
        config_path = list(Path(root_path).rglob('*.yaml'))[0]

        config = OmegaConf.load(config_path)
        self.is_overide = False
        self.test_name = Path(weight_path).stem
        
        if len(overide_params) > 0:
            if len(overide_params) > 0:
                overide_config = OmegaConf.from_cli(overide_params)
                config = OmegaConf.merge(config, overide_config)
                self.test_name = Path(weight_path).stem + '_'.join([overide_p.split('=')[-1].split('/')[-1] if len(overide_p.split('=')[-1].split('/')) > 1 else overide_p.split('=')[-1] for overide_p in overide_params])
                self.is_overide = True
            
        self.config = config
        
        test_dataset = initialize_from_config(config.testdataset)
        self.random_style_image=test_dataset.random_style_image

        self.test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batchsize,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=4,
                        collate_fn=test_dataset.collate_function if hasattr(self.config.training, 'use_collate_fn') and self.config.training.use_collate_fn else None
                    )
        
        if not hasattr(self.config.text_encoder.params, 'input_size'):
            self.config.text_encoder.params.input_size = test_dataset.tokenizer.vocab_size
        
        pipeline = get_obj_from_str(self.config.pipeline.target)(
            vae_config=self.config.vae.vae_pretrained_path,
            diffusion_config=self.config.diffusion,
            style_extractor_config=self.config.style_extractor,
            text_encoder_config=self.config.text_encoder
        )
        pipeline.load_state_dict(weight_path)
        pipeline.eval()
        self.pipeline = pipeline.to(self.device)
        
    def setup_logger(self, weight_path):
        root_dir = Path(weight_path).joinpath('eval', self.test_name)
        make_dirs(root_dir)
        image_dir = root_dir.joinpath('evaluate_images')
        self.real_dir = image_dir.joinpath('real')
        make_dirs(self.real_dir)
        self.gen_dir = image_dir.joinpath('gen')
        make_dirs(self.gen_dir)
        # visualize purpose only
        self.pair_dir = image_dir.joinpath('pair')
        make_dirs(self.pair_dir)
        logger.add(root_dir.joinpath('test.log'), format="{time} {level} {message}", level="INFO")
        if self.is_overide:
        #Save current config information
            with open(root_dir.joinpath('overide.yaml'), 'w') as f:
                yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)
                
    def eval(self):
        img_idx = 0
        start_time = time.time()

        for batch in self.test_loader:
            style_images, text_embeddings, ori_wids, image_names = batch['style_images'], batch['text_embeddings'], batch['ori_wids'], batch['image_names']
            original_lens = batch['original_lens']

            if self.random_style_image:
                style_names = batch['style_names']
                if self.save_predefined_json:
                    for img_name, style_name in zip(image_names, style_names):
                        predefined_info[img_name]=style_name

            skip=False
            for ori_wid, image_name in zip(ori_wids, image_names):
                if self.real_dir.joinpath(str(ori_wid)).joinpath(image_name).exists():
                    skip=True
                    break
            if skip:
                continue

            target_images = batch['images'].to(self.device)
            style_images = style_images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            gen_images = self.pipeline.sampling(original_lens, text_embeddings, style_images, log_progress=False)
            target_images = torch.unbind(target_images, dim=0)

            new_target_images = []
            for img, img_len in zip(target_images, original_lens):
                new_target_images.append(img[:, :, :img_len])
            target_images=new_target_images

            style_images = torch.unbind(style_images, dim=0)
            if not isinstance(gen_images, list):
                gen_images = torch.unbind(gen_images, dim=0)
        
            for img, ref_img, gen_img, ori_wid in zip(target_images, style_images, gen_images, ori_wids):
                img = (img + 1)*127.5
                img = img.permute(1, 2, 0).cpu().numpy().astype('uint8')
                img = Image.fromarray(img)
                by_wid_ori = self.real_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_ori)
                img_name = image_names.pop(0)
                if len(img_name.split('/')) > 1:
                    img_name = img_name.split('/')[-1]
                img.save(by_wid_ori.joinpath(img_name))

                gen_img = gen_img*255.0
                gen_img = gen_img.permute(1, 2, 0).cpu().numpy().astype('uint8')
                gen_img = Image.fromarray(gen_img)
                by_wid_gen = self.gen_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_gen)
                if self.save_by_name:
                    gen_img.save(by_wid_gen.joinpath(img_name))
                else:
                    gen_img.save(by_wid_gen.joinpath(str(img_idx) + '.png'))
                ref_img = ref_img*255.0
                ref_img = ref_img.permute(1, 2, 0).cpu().numpy().astype('uint8')
                ref_img = Image.fromarray(ref_img)
                by_wid_pair = self.pair_dir.joinpath(str(ori_wid))
                make_dirs(by_wid_pair)
                pair_img = make_image_grid([img, gen_img, ref_img], rows=1, cols=3, maxsize=256)
                pair_img.save(by_wid_pair.joinpath(str(img_idx) + '.png'))
                
                img_idx += 1
        
        elapsed_time = (time.time() - start_time)
        min, sec = divmod(elapsed_time, 60)
        logger.info('Complete sampling with total number of samples', img_idx)
        logger.info('Elapsed time: {} m - {} s', int(min), round(sec, 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default='/data/ocr/duyla4/Research/Diffusion/hand_gen/ACCV2024/output/final_method/flow_by_iteration/combine_all/base_quantized_downsample1_1024num768embed_detach_stylesymmetricweight01contentbidirectional_lossbalnacing_contrastive_attnpoolconcatrefine_longerver_variable/20250110021042') #/data/ocr/duyla4/Research/Diffusion/hand_gen/ACCV2024/output/final_method/flow_by_iteration/combine_all/base_quantized_downsample1_1024num768embed_detach_stylesymmetricweight01contentbidirectional_lossbalnacing_contrastive_attnpoolconcatrefine_longerver_variable/20250110021042
    parser.add_argument('--weight-path', default='/data/ocr/duyla4/Research/Diffusion/hand_gen/ACCV2024/output/final_method/flow_by_iteration/combine_all/base_quantized_downsample1_1024num768embed_detach_stylesymmetricweight01contentbidirectional_lossbalnacing_contrastive_attnpoolconcatrefine_longerver_variable/20250110021042/checkpoint/iter_650000.pth')
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--cond_scale', default=None)
    parser.add_argument('--rescaled_phi', default=None)
    parser.add_argument('--sampling_steps', default=None)
    parser.add_argument('--ddim_sampling', default=None)
    parser.add_argument('--random_style', default=False)
    parser.add_argument('--image_path', default=None) #'dataset/IAM_variable_size/rescale_images'
    parser.add_argument('--random_style_image', default=False)
    parser.add_argument('--writer_dict_path', default='writers_dict_test.json') #writers_dict_test
    parser.add_argument('--full_dict_path', default='IAM_longver_test_transcriptions.json') #IAM_longver_test_transcriptions
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    set_seed(0)
    
    overide_params = [
        f'diffusion.params.clfree_params.cond_scale={args.cond_scale}' if args.cond_scale is not None else None,
        f'diffusion.params.clfree_params.rescaled_phi={args.rescaled_phi}'  if args.rescaled_phi is not None else None,
        f'diffusion.params.sampling_steps={args.sampling_steps}' if args.sampling_steps is not None else None,
        f'diffusion.params.ddim_sampling={args.ddim_sampling}' if args.ddim_sampling is not None else None,
        f'testdataset.params.random_style={args.random_style}' if args.random_style is not None else None,
        f'testdataset.params.random_style_image={args.random_style_image}' if args.random_style_image is not None else None,
        f'testdataset.params.writer_dict_path={args.writer_dict_path}' if args.writer_dict_path is not None else None,
        f'testdataset.params.full_dict_path={args.full_dict_path}' if args.full_dict_path is not None else None,
        f'testdataset.params.image_path={args.image_path}' if args.image_path is not None else None
    ]
    overide_params = list(filter(lambda p: p is not None, overide_params))
    runner = Runner(args.device, args.root_path, args.weight_path, args.batchsize, overide_params)
    runner.setup_logger(args.root_path)
    runner.eval()
