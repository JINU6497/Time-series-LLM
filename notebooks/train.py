import math
import os
import time

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from PIL import Image

import wandb
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline


logger = get_logger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def training(cfg, accelerator, tokenizer, placeholder_token_id, text_encoder,
             vae, unet, noise_scheduler, optimizer, lr_scheduler, train_loader, test_loader):

    # Logging Setting
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    # gradient_checkpointing
    if cfg.TRAIN.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # Weight Data Type (Mixed Precision)
    weight_dtype = torch.float32
    match accelerator.mixed_precision:
        case "fp16":
            weight_dtype = torch.float16
        case "bf16":
            weight_dtype = torch.bfloat16

    # Move vae and unet to device, and convert its weights to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    if cfg.TRAIN.gradient_checkpointing:
        unet.train()
    else:
        unet.eval()

    # Recalculate Training Steps
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.TRAIN.gradient_accumulation_steps)
    num_train_epochs = math.ceil(
        cfg.TRAIN.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = cfg.DATALOADER.batch_size * \
        accelerator.num_processes * cfg.TRAIN.gradient_accumulation_steps

    logger.info(
        f"***** Running Training for total {num_train_epochs} epochs *****")
    logger.info(f"  Num examples = {cfg.TRAIN.train_set_length}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.DATALOADER.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.TRAIN.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.TRAIN.max_train_steps}")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(
        text_encoder).get_input_embeddings().weight.data.clone()
    step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        end = time.time()
        for idx, (img, mask, target, input_ids) in enumerate(train_loader):
            with accelerator.accumulate(text_encoder):
                data_time_m.update(time.time() - end)
                # Convert images to latent space
                latents = vae.encode(
                    img.to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image (timestep should be betweeen 0 ~ 100)
                timesteps = torch.randint(
                    cfg.TRAIN.time_steps[0], cfg.TRAIN.time_steps[1], (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    input_ids)[0].to(dtype=weight_dtype)

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states.to(weight_dtype)).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(),
                                  target.float(), reduction="mean")
                accelerator.backward(loss)

                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones(
                    (len(tokenizer),), dtype=torch.bool)
                # index_no_updates[min(placeholder_token_id): max(placeholder_token_id) + 1] = False
                index_no_updates[(placeholder_token_id)] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                step += 1
                # log loss
                losses_m.update(loss.item())
                batch_time_m.update(time.time() - end)
                if (step) % cfg.LOG.log_interval == 0:
                    logger.info('TRAIN [{:>4d}/{}] '
                                'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'LR: {lr:.6f} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                    step+1, cfg.TRAIN.max_train_steps,
                                    loss=losses_m,
                                    lr=optimizer.param_groups[0]['lr'],
                                    batch_time=batch_time_m,
                                    rate=img.size(0) / batch_time_m.val,
                                    rate_avg=img.size(0) / batch_time_m.avg,
                                    data_time=data_time_m))
                    accelerator.log(
                        {"lr": optimizer.param_groups[0]["lr"], "MSE_LOSS": losses_m.val, }, step=step,)
                if (step) % cfg.LOG.eval_interval == 0:
                    # Evaluate 시 main process에서만 진행
                    if accelerator.is_main_process:
                        evaluate(text_encoder, tokenizer, unet, vae, cfg,
                                 accelerator, weight_dtype, noise_scheduler, step, test_loader)

                if step % cfg.LOG.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        weight_name = "learned_embeds.bin" if cfg.RESULT.no_safe_serialization else "learned_embeds.safetensors"
                        save_path = os.path.join(
                            cfg.RESULT.savedir, weight_name)
                        logger.info(f"Saving embeddings to {save_path}")
                        save_progress(
                            text_encoder,
                            placeholder_token_id,
                            accelerator,
                            cfg,
                            save_path,
                            safe_serialization=not cfg.RESULT.no_safe_serialization,
                        )
            if step >= cfg.TRAIN.max_train_steps:
                break
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.MODEL.pretrained,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(cfg.RESULT.savedir)
        logger.info(f"Saved pipeline to {cfg.RESULT.savedir}")
        save_path = os.path.join(cfg.RESULT.savedir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id,
                      accelerator, cfg, save_path, True)
        logger.info("Saving embeddings")
        logger.info(f"Saved embeddings to {save_path}")
    # end of training
    logger.info("Training is finished")
    accelerator.end_training()


def evaluate(text_encoder, tokenizer, unet, vae, cfg, accelerator, weight_dtype, noise_scheduler, step, test_loader):
    logger.info(
        "Running evaluation \n Generating reconstruction images in test_loader")
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        cfg.MODEL.name,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = None if cfg.SEED is None else torch.Generator(
        device=accelerator.device).manual_seed(cfg.SEED)

    origin_imgs, masks, targets, recon_imgs = [], [], [], []
    for idx, (img, mask, target, input_ids) in enumerate(test_loader):
        with torch.autocast("cuda"):
            bsz = img.shape[0]
            # Convert images to latent space
            latents = vae.encode(
                img.to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                cfg.VALIDATION.time_steps[0], cfg.VALIDATION.time_steps[1], (bsz,), device=latents.device).long()

            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)
            for i in range(bsz):
                recon_imgs.append(np.array(pipeline([cfg.VALIDATION.validation_prompt] * cfg.VALIDATION.num_validation_images,  num_inference_steps=cfg.VALIDATION.inversion_steps,
                                                    guidance_scale=cfg.VALIDATION.guidance_scale, latents=noisy_latents[i:i+1, :, :, :], generator=generator).images[0]))

            origin_imgs.extend(
                list(((img.cpu().numpy()+1.0)*127.5).astype(np.uint8)))
            masks.extend(list(mask.cpu().numpy()))
            targets.extend(list(target.cpu().numpy()))
            # todo 여기부터 계속 하면 됨! (Reconstruction Based Anomaly Detection & Attention Map Based Anomaly Detection)

    accelerator.log(
        {
            "Original Image": [
                wandb.Image(Image.fromarray(image.transpose(1, 2, 0))) for _, image in enumerate(origin_imgs)
            ],
            "Reconstruction Image": [
                wandb.Image(Image.fromarray(image)) for _, image in enumerate(recon_imgs)
            ],
            "Ground Truth Mask": [
                # TODO mask 부분 뭔가 이상함 다 안 뽑힘
                wandb.Image(Image.fromarray(image*255, 'L')) for _, image in enumerate(masks)
            ],
            #
        }, step=step)
    del pipeline
    torch.cuda.empty_cache()


def save_progress(text_encoder, placeholder_token_id, accelerator, cfg, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        #.weight[min(placeholder_token_ids): max(placeholder_token_ids) + 1]
        .weight[(placeholder_token_id)]
    )
    learned_embeds_dict = {
        cfg.TRAIN.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)
