import copy
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchinfo import summary
from torchvision.models import swin_s, Swin_S_Weights, maxvit_t, MaxVit_T_Weights
from torch.amp import GradScaler
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from .utils import load_model 

def trainer(
    model: nn.Module,
    train_batch,
    val_batch,
    num_epochs,
    lr,
    model_config,
    device = "cuda",
    early_stop=5,
    resume=False,
    out_file=None,
    num_classes=16
):
    loss_scaler = GradScaler('cuda')

    summary(model, input_size=(model_config.batch_size, 3, 224, 224))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=model_config.weight_decay)
    scheduler = create_scheduler(model_config.scheduler, optimizer, lr, num_epochs, len(train_batch))
    
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = SoftTargetCrossEntropy()
    
    mixup_fn = Mixup(
            mixup_alpha=model_config.mixup, cutmix_alpha=model_config.cutmix, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=num_classes)

    model = model.to(device)
    model.train()

    best_accuracy = 0
    stop_increasing = 0

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    start_epoch = 0
    if resume:
        print("Resume Training from previous check point")
        start_epoch = load_model(model, optimizer, loss_scaler, out_file)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        progress_bar = tqdm(train_batch, 
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       leave=True)
        current_losses = []
        for batch in progress_bar:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            X, y = mixup_fn(X, y)
            # --mixup 0.8 --cutmix 1.0
            with torch.autocast(device_type=device):
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                
                loss_history.append(loss.item())
                current_losses.append(loss.item())

            loss_scaler.scale(loss).backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            loss_scaler.step(optimizer)
            loss_scaler.update()  

            if model_config.scheduler == "OneCycleLR":
                scheduler.step()

        train_accuracy = calc_accuracy(model, train_batch, device=device)
        val_accuracy = calc_accuracy(model, val_batch, device=device)
        
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        avg_loss = sum(current_losses) / len(current_losses)

        if model_config.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_accuracy)

        if epoch % model_config.log_interval == 0: 
            print("Learning Rate:", scheduler.get_last_lr())
            print(f"Epoch {epoch+1} Loss: {avg_loss} Train Accuracy: {train_accuracy} \
            Validation Accuracy: {val_accuracy}")    

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f'Global gradient norm: {total_norm}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stop_increasing = 0
        else:
            stop_increasing += 1

        if stop_increasing == early_stop:
            print("Train Early Stop...")
            break
    
    # if best_params is not None:
    #     model.load_state_dict(best_params)
    
    return loss_history, train_accuracy_history, \
    val_accuracy_history, best_accuracy, model, optimizer, loss_scaler, start_epoch + num_epochs - 1

def create_model(model_name, num_classes):
    if model_name == "swin":
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == "maxvit":
        model = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        
        model.classifier[-1] = nn.Linear(
            in_features=in_features,
            out_features=num_classes
        )
    return model

def create_scheduler(scheduler_name, optimizer, lr, num_epochs, n_train_batch):
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.2, mode='max')
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=n_train_batch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
    return scheduler

def calc_accuracy(model, batchs, device="cpu"):
    model.eval()
    
    corrects = 0
    total = 0
    with torch.no_grad():
        for batch in batchs:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            corrects += pred.argmax(dim=1).eq(y).sum().item()
            total += len(y)

    return corrects / total