import argparse
import datetime
import logging
import math

import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from pytz import timezone
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from data import create_dataloader, create_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from train import training


if is_wandb_available():
    pass

logger = get_logger(__name__)


def tokenizer_setting(pretrained_model: str, placeholder_token: str, initializer_token: str):
    # Tokenizer Setting
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model, subfolder="tokenizer", )

    # Add the placeholder token in tokenizer #special token 매핑
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                         " `placeholder_token` that is not already in the tokenizer.")
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    return tokenizer, placeholder_token_id, initializer_token_id


def run(cfg):
    # Set Seed
    set_seed(cfg.SEED)
    # Init Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb",
                              kwargs_handlers=[ddp_kwargs],
                              gradient_accumulation_steps=cfg.TRAIN.gradient_accumulation_steps,
                              mixed_precision=cfg.TRAIN.mixed_precision,
                              )

    # logging setting
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Model Setting
    if cfg.MODEL.pretrained:
        # tokenizer setting
        tokenizer, placeholder_token_id, initializer_token_id = tokenizer_setting(
            cfg.MODEL.name, cfg.TRAIN.placeholder_token, cfg.TRAIN.initializer_token)
        # text_encoder setting (CLIP)
        text_encoder = CLIPTextModel.from_pretrained(
            cfg.MODEL.name, subfolder="text_encoder")
    else:
        pass  # TODO: PRETRIAINED FALSE 인경우도 제작

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

    # Autoencoder Setting (VAE)
    vae = AutoencoderKL.from_pretrained(cfg.MODEL.name, subfolder="vae")
    # Diffusion Model Setting(UNet)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.MODEL.name, subfolder="unet")
    # Scheduler Setting (DDPM)
    noise_scheduler = DDPMScheduler.from_config(
        cfg.MODEL.name, subfolder="scheduler")

    # Follow the size of vae.sample_size
    cfg.DATASET.size = vae.sample_size

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if cfg.TRAIN.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # Dataset Setting
    train_dataset = create_dataset(dataset_name=cfg.DATASET.name,
                                   data_dir=cfg.DATASET.datadir,
                                   target=cfg.DATASET.target,
                                   is_train=True,
                                   tokenizer=tokenizer,
                                   learnable_property=cfg.TRAIN.learnable_property,
                                   size=cfg.DATASET.size,
                                   placeholder_token=cfg.TRAIN.placeholder_token,
                                   few_shot=cfg.DATASET.fewshot,
                                   few_shot_num=cfg.DATASET.fewshot_num,
                                   )
    test_dataset = create_dataset(dataset_name=cfg.DATASET.name,
                                  data_dir=cfg.DATASET.datadir,
                                  target=cfg.DATASET.target,
                                  is_train=False,
                                  tokenizer=tokenizer,
                                  learnable_property=cfg.TRAIN.learnable_property,
                                  size=cfg.DATASET.size,
                                  placeholder_token=cfg.VALIDATION.validation_prompt,
                                  few_shot=cfg.DATASET.fewshot,
                                  few_shot_num=cfg.DATASET.fewshot_num,
                                  )

    # Dataloader Setting
    train_loader = create_dataloader(train_dataset,
                                     shuffle=False,
                                     batch_size=cfg.DATALOADER.batch_size,
                                     num_workers=cfg.DATALOADER.num_workers)

    test_loader = create_dataloader(test_dataset,
                                    shuffle=False,
                                    batch_size=cfg.DATALOADER.batch_size,
                                    num_workers=cfg.DATALOADER.num_workers)

    cfg.TRAIN.train_set_length = len(train_dataset)
    # Training Setting
    # Learning Rate Setting
    learning_rate = cfg.OPTIMIZER.lr
    if cfg.OPTIMIZER.scale_lr:
        learning_rate *= accelerator.num_processes * cfg.gradient_accumulation_steps

    # Optimizer Setting
    optimizer = torch.optim.AdamW(
        # only optimize the textembeddings
        text_encoder.get_input_embeddings().parameters(),
        lr=learning_rate,
        betas=(cfg.LOSS.adam_beta1, cfg.LOSS.adam_beta2),
        weight_decay=cfg.OPTIMIZER.weight_decay,
        eps=cfg.LOSS.adam_epsilon,
    )
    # Learning Rate Scheduler Setting
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.TRAIN.gradient_accumulation_steps)
    if cfg.TRAIN.max_train_steps is None:
        cfg.TRAIN.max_train_steps = cfg.TRAIN.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.OPTIMIZER.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.TRAIN.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.OPTIMIZER.lr_num_cycles,
    )

    # time (KST)
    current_time = datetime.datetime.now(
        timezone('Asia/Seoul')).strftime('%Y.%m.%d.%H:%M')
    # Tracker Init
    configs_ = {'Dataset': cfg.DATASET.name,
                'Target': cfg.DATASET.target,
                'Fewshot': cfg.DATASET.fewshot,
                'Fewshot_num': cfg.DATASET.fewshot_num,
                'Image_size': cfg.DATASET.size,
                'Dataset_Repeats': cfg.DATASET.repeats,
                'Pretrained_model': cfg.MODEL.name,
                'Train_Steps': cfg.TRAIN.max_train_steps,
                'Mixed_Precision': cfg.TRAIN.mixed_precision,
                'Placeholder_token': cfg.TRAIN.placeholder_token,
                'Initializer_token': cfg.TRAIN.initializer_token,
                'Forward_Timesteps': cfg.TRAIN.time_steps,
                'Validation_Prompt': cfg.VALIDATION.validation_prompt,
                'Validation_Steps': cfg.VALIDATION.inversion_steps,
                'Backward_Timesteps': cfg.VALIDATION.time_steps,
                'Guidance_Scale': cfg.VALIDATION.guidance_scale,
                'Learning_Rate': cfg.OPTIMIZER.lr,
                'LR_Scheduler': cfg.OPTIMIZER.lr_scheduler,
                'Weight_Decay': cfg.OPTIMIZER.weight_decay,}
    
    accelerator.init_trackers(project_name=cfg.EXP_NAME,
                              config=configs_,
                              init_kwargs={"wandb":
                                           {'entity': 'vision_brother',
                                            # name (DATAEST TARGET + _yyyy.mm.dd.hh)
                                            'name': cfg.DATASET.target + '_' + current_time, }
                                           }
                              )

    # accelrate prepare
    text_encoder, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_loader, test_loader, lr_scheduler
    )

    training(cfg=cfg,
             accelerator=accelerator,
             tokenizer=tokenizer,
             placeholder_token_id=placeholder_token_id,
             text_encoder=text_encoder,
             vae=vae,
             unet=unet,
             noise_scheduler=noise_scheduler,
             optimizer=optimizer,
             lr_scheduler=lr_scheduler,
             train_loader=train_loader,
             test_loader=test_loader,
             )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Textual Inversion Anomaly Detection')
    parser.add_argument('--configs', type=str,
                        default=None, help='exp config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    # load cfg
    cfg = OmegaConf.load(args.configs)

    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        try:
            OmegaConf.update(cfg, k, eval(v), merge=True)
        except:
            OmegaConf.update(cfg, k, v, merge=True)

    print(OmegaConf.to_yaml(cfg))

    run(cfg)
