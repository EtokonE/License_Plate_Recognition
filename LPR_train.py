import argparse
from pathlib import Path
import sys
import os
import logging
import time
import torch
import torch.nn as nn
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer
from src.config.config import combine_config
from src.data.LPR_dataset import LPRDataset, collate_fn
from src.tools.utils import colorstr, decode_function
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_root_path() -> Path:
    """Get relative root dir"""
    curr_file = Path(__file__).resolve()
    root_dir = curr_file.parents[0]
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    root_dir = Path(os.path.relpath(root_dir, Path.cwd()))
    return root_dir


def sparse_tuple_for_ctc(predicted_length: int, gt_lengths: List[int]) -> (tuple, tuple):
    """Construct matched tuples for ctc loss calculation

    Args:
        predicted_length (int): LPRNet output shape (The initially defined length of the predicted sequence)
        gt_lengths (list[int]): list of ground truth labels length for current batch in LPRNet dataloader

    Returns:
        input_lengths (tuple): tuple, each element of which corresponds to the length of LPRNet output
        target_lengths (tuple): tuple, each element of which corresponds ground truth label length
    """
    input_lengths = []
    target_lengths = []

    for gt_len in gt_lengths:
        input_lengths.append(predicted_length)
        target_lengths.append(gt_len)

    return tuple(input_lengths), tuple(target_lengths)


def create_parser():
    parser = argparse.ArgumentParser(description='Parameters to train LPRNet with ST module')
    parser.add_argument('--out_dir', type=str, help='Directory to save results')
    parser.add_argument('--config', type=str, help='Path to experiment config')
    args = parser.parse_args()
    return args


def get_final_config():
    """Compare config and command line args"""
    args = create_parser()
    cfg = combine_config(args.config)
    cfg.LPRNet.TRAIN.OUT_FOLDER = args.out_dir
    cfg.ROOT.PATH = str(get_root_path())
    return cfg


def save_final_config(config, save_path):
    config.dump(stream=open(os.path.join(save_path, f'config.yaml'), 'w'))


def create_out_folder(output_path: str) -> None:
    """Create folder to store train results"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    weights_path = os.path.join(output_path, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)


def get_logger(logfile="debug.log", stdout=True):
    handlers = [logging.FileHandler(logfile)]
    if stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="\n%(asctime)s [%(levelname)s]\n%(message)s",
        handlers=handlers
    )
    return logging


def create_log_template():
    """Create template to save logs"""
    return "Epoch: {ep:03d}/{full_ep:03d}, train_loss: {t_loss:0.4f}, \
        train_accuracy {t_acc:0.4f}, val_loss: {v_loss:0.4f}, val_accuracy {v_acc:0.4f}, \
        time: {time:.2f} s/iter, learning rate: {lr:0.4f}\n"


def weights_initialization(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


def initialize_lprnet_weights(lpr_model):
    """Initialize weights for LPR net"""
    lpr_model.backbone.apply(weights_initialization)
    lpr_model.container.apply(weights_initialization)
    print('Successful init LPR weights')


def load_lpr_weights(model, weights, device='cpu'):
    """Load pretrained weights for LPR net"""
    model_sd = model.state_dict()
    pretrained_model = torch.load(weights, map_location=torch.device(device))
    filtered_dict = {k: v for k, v in pretrained_model.items() if k in model_sd}
    filtered_dict.pop("backbone.20.weight")
    filtered_dict.pop("backbone.20.bias")
    filtered_dict.pop("backbone.21.weight")
    filtered_dict.pop("backbone.21.bias")
    filtered_dict.pop("backbone.21.running_mean")
    filtered_dict.pop("backbone.21.running_var")
    model_sd.update(filtered_dict)
    model.load_state_dict(model_sd)
    print(f'Successful load weights for model: {model}')

def load_stm_weights(model, weights, device='cpu'):
    return model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

def build_lprnet(config, device):
    lprnet = LPRNet(class_num=len(config.CHARS.LIST),
                    dropout_prob=config.LPRNet.DROPOUT,
                    out_indices=config.LPRNet.OUT_INDEXES)
    lprnet.to(device)
    # Init LPR net weights
    if config.LPRNet.TRAIN.PRETRAINED_MODEL:
        load_lpr_weights(model=lprnet, weights=config.LPRNet.TRAIN.PRETRAINED_MODEL, device=device)
    else:
        initialize_lprnet_weights(lprnet)
    return lprnet


def build_stn(config, device):
    stn = SpatialTransformer()
    stn.to(device)
    # Init Spatial Transformer weights
    if config.LPRNet.TRAIN.PRETRAINED_SPATIAL_TRANSFORMER:
        load_stm_weights(model=stn, weights=config.LPRNet.TRAIN.PRETRAINED_SPATIAL_TRANSFORMER, device=device)
    return stn


def create_datasets(cfg) -> dict:
    datasets = {
        'train': LPRDataset(
            data_dir=cfg.LPR_dataset.TRAIN_PATH,
            chars=cfg.CHARS.LIST,
            mode='train',
            img_size=cfg.LPRNet.TRAIN.IMG_SIZE
        ),
        'val': LPRDataset(
            data_dir=cfg.LPR_dataset.VAL_PATH,
            chars=cfg.CHARS.LIST,
            mode='val',
            img_size=cfg.LPRNet.TRAIN.IMG_SIZE
        )
    }
    return datasets


def create_dataloaders(datasets: dict, cfg) -> dict:
    dataloaders = {
        'train': DataLoader(
            dataset=datasets['train'],
            batch_size=cfg.LPRNet.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.LPRNet.TRAIN.NUM_WORKERS,
            collate_fn=collate_fn
        ),
        'val': DataLoader(
            dataset=datasets['val'],
            batch_size=cfg.LPRNet.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.LPRNet.TRAIN.NUM_WORKERS,
            collate_fn=collate_fn
        )
    }
    print('Dataloaders created!')
    return dataloaders


def fit_epoch(lpr_model, spatial_transformer_model,
              train_loader, criterion, optimizer,
              device, chars, decode_fn, predicted_length):

    spatial_transformer_model.train()
    lpr_model.train()

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for imgs, labels, lengths in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            transfer = spatial_transformer_model(imgs)
            logits = lpr_model(transfer)  # torch.Size([batch_size, CHARS length, output length ])
            log_probs = logits.permute(2, 0, 1)  # for ctc loss: output length x batch_size x CHARS length
            log_probs = log_probs.log_softmax(2).requires_grad_()
            ctc_input_lengths, ctc_target_lengths = sparse_tuple_for_ctc(predicted_length=predicted_length,
                                                                         gt_lengths=lengths)
            loss = criterion(
                log_probs,
                labels,
                input_lengths=ctc_input_lengths,
                target_lengths=ctc_target_lengths
            )

            loss.backward()
            optimizer.step()

            preds = logits.cpu().detach().numpy()
            _, pred_labels = decode_fn(preds, chars)
            print(_, pred_labels)
            start = 0
            true_positive = 0
            for i, length in enumerate(lengths):
                label = labels[start:start + length]
                print(label)
                start += length
                if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                    true_positive += 1

        running_loss += loss.item() * imgs.size(0)
        running_corrects += true_positive
        processed_data += imgs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects / processed_data
    return train_loss, train_acc,


def eval_epoch(lpr_model, spatial_transformer_model,
               val_loader, criterion, decode_fn,
               device, chars, predicted_length):

    lpr_model.eval()
    spatial_transformer_model.eval()

    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for imgs, labels, lenghts in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            transfer = spatial_transformer_model(imgs)
            logits = lpr_model(transfer)
            log_probs = logits.permute(2, 0, 1)  # for ctc loss: output length x batch_size x CHARS length
            log_probs = log_probs.log_softmax(2).requires_grad_()
            ctc_input_lengths, ctc_target_lengths = sparse_tuple_for_ctc(predicted_length=predicted_length,
                                                                         gt_lengths=lenghts)
            loss = criterion(
                log_probs,
                labels,
                input_lengths=ctc_input_lengths,
                target_lengths=ctc_target_lengths
            )
            preds = logits.cpu().detach().numpy()
            _, pred_labels = decode_fn(preds, chars)

            start = 0
            true_positive = 0
            for i, length in enumerate(lenghts):
                label = labels[start:start + length]
                start += length
                if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                    true_positive += 1

        running_loss += loss.item() * imgs.size(0)
        running_corrects += true_positive
        processed_size += imgs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects / processed_size
    return val_loss, val_acc


def train():
    # Define configuration file
    cfg = get_final_config()

    # Define logger
    create_out_folder(cfg.LPRNet.TRAIN.OUT_FOLDER)
    logger = get_logger(logfile=os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'train_logs.log'))
    logger.info(colorstr('hyperparameters: \n') +
                '\n'.join(f'{colorstr("cyan", k)}\n{v}\n--------' for k, v in cfg.items()))
    save_final_config(config=cfg, save_path=cfg.LPRNet.TRAIN.OUT_FOLDER)

    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(colorstr('yellow', 'bold', 'Train on device: ') + f'{device}')

    # Create models
    lprnet = build_lprnet(cfg, device)
    stn = build_stn(cfg, device)

    # Load data
    datasets = create_datasets(cfg)
    dataloaders = create_dataloaders(datasets, cfg)
    logger.info(colorstr('yellow', 'bold', 'Training dataset loaded successfully.') +
                f'Total training examples: {len(datasets["train"])}')
    logger.info(colorstr('yellow', 'bold', 'Validation dataset loaded successfully.') +
                f'Total validation examples: {len(datasets["val"])}')

    # -------------------
    # Train
    # -------------------
    start_time = time.time()
    best_acc = 0.0
    log_template = create_log_template()
    history = []
    print('-' * 10)

    with tqdm(desc="epoch", total=cfg.LPRNet.TRAIN.NUM_EPOCHS) as pbar_outer:
        # Define optimizer & loss & sheduler
        optimizer = torch.optim.Adam([{'params': stn.parameters(), 'weight_decay': 2e-5},
                                      {'params': lprnet.parameters()}],
                                     lr=cfg.LPRNet.TRAIN.LR)
        ctc_loss = nn.CTCLoss(blank=len(cfg.CHARS.LIST) - 1, reduction='mean')
        lr_sheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.LPRNet.TRAIN.LR_SHED_GAMMA)

        for epoch in range(cfg.LPRNet.TRAIN.NUM_EPOCHS):
            start_time_ep = time.time()
            train_loss, train_acc = fit_epoch(
                lpr_model=lprnet,
                spatial_transformer_model=stn,
                train_loader=dataloaders['train'],
                criterion=ctc_loss,
                optimizer=optimizer,
                device=device,
                chars=cfg.CHARS.LIST,
                decode_fn=decode_function,
                predicted_length=cfg.LPRNet.PREDICTED_LENGTHS
            )

            val_loss, val_acc = eval_epoch(
                lpr_model=lprnet,
                spatial_transformer_model=stn,
                val_loader=dataloaders['val'],
                criterion=ctc_loss,
                device=device,
                chars=cfg.CHARS.LIST,
                decode_fn=decode_function,
                predicted_length=cfg.LPRNet.PREDICTED_LENGTHS
            )

            pbar_outer.update(1)
            if (epoch + 1) % cfg.LPRNet.TRAIN.SAVE_PERIOD == 0:
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': lprnet.state_dict()},
                    os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'weights', f'lprnet_Ep_{epoch+1}_model.ckpt'))

                torch.save({
                    'epoch': epoch,
                    'net_state_dict': stn.state_dict()},
                    os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'weights', f'stn_Ep_{epoch+1}_model.ckpt'))

            for p in optimizer.param_groups:
                curr_lr = p['lr']

            history.append((train_loss, train_acc, val_loss, val_acc, curr_lr))

            if (epoch > cfg.LPRNet.TRAIN.NUM_EPOCHS / 4) and (curr_lr > 0.0003):
                lr_sheduler.step()

            end_time_ep = time.time()
            msg = log_template.format(ep=epoch + 1, full_ep=cfg.LPRNet.TRAIN.NUM_EPOCHS, t_loss=train_loss,
                                      t_acc=train_acc, v_loss=val_loss, v_acc=val_acc,
                                      time=end_time_ep - start_time_ep, lr=curr_lr)

            logger.info(msg)

            if val_acc >= best_acc:
                best_acc = val_acc
                best_ep = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': lprnet.state_dict()},
                    os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'weights', f'lprnet_BEST_model.ckpt'))

                torch.save({
                    'epoch': epoch,
                    'net_state_dict': stn.state_dict()},
                    os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'weights', f'stn_BEST_model.ckpt'))

    time_elapsed = time.time() - start_time
    logger.info('Finally Best Accuracy: {:.4f} in epoch: {}'.format(best_acc, best_ep))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    history_df = pd.DataFrame(history, columns=[
                                                'train_loss',
                                                'train_acc',
                                                'val_loss',
                                                'val_acc',
                                                'lr'
                                                ])
    history_df.to_csv(os.path.join(cfg.LPRNet.TRAIN.OUT_FOLDER, 'history.csv'))


if __name__ == '__main__':
    train()
