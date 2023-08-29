# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
import tableprint as tp

import torch
import torchnet as tnt


def run_epoch(dataloader,
              loader_size,
              model,
              teacher_model,
              criterion,
              aux_criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              logger,
              scaler,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    aux_loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L40
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext

    teacher_model.eval()
    with torch.set_grad_enabled(True), model_context():
        for i, batch in enumerate(dataloader):
            utts = batch['key']
            targets = batch['label']
            features = batch['feat']
            wav = batch['wav']

            cur_iter = (epoch - 1) * loader_size + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            features = features.float().to(device)  # (B,T,F)
            targets = targets.long().to(device)

            teacher_model.eval()
            with torch.cuda.amp.autocast(enabled=enable_amp):
                outputs = model(features)  # (embed_a,embed_b) in most cases
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = model.module.projection(embeds, targets)
                with torch.no_grad():
                    teacher_embeds = teacher_model(wav)
                main_loss = criterion(outputs, targets)
                aux_loss = aux_criterion(embeds, teacher_embeds)
                loss = main_loss + aux_loss
            # loss, acc
            loss_meter.add(main_loss.item())
            aux_loss_meter.add(aux_loss.item())
            acc_meter.add(outputs.cpu().detach().numpy(),
                          targets.cpu().numpy())

            # updata the model
            optimizer.zero_grad()
            # scaler does nothing here if enable_amp=False
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log
            if (i + 1) % log_batch_interval == 0:
                logger.info(
                    tp.row((epoch, i + 1, scheduler.get_lr(),
                            margin_scheduler.get_margin()) +
                           (loss_meter.value()[0], round(aux_loss_meter.value()[0], 7), acc_meter.value()[0]),
                           width=10,
                           style='grid'))

    logger.info(
        tp.row((epoch, i + 1, scheduler.get_lr(),
                margin_scheduler.get_margin()) +
               (loss_meter.value()[0], round(aux_loss_meter.value()[0], 7), acc_meter.value()[0]),
               width=10,
               style='grid'))
