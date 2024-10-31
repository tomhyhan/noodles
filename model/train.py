import copy
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchinfo import summary
from torchvision.models import swin_s, Swin_S_Weights

def trainer(
    model: nn.Module,
    train_batch,
    val_batch,
    num_epochs,
    lr,
    batch_size,
    weight_decay,
    early_stop=5,
    log_interval=1,
    scheduler_name="ReduceLROnPlateau",
    device = "cpu"
):
    
    summary(model, input_size=(batch_size, 3, 224, 224))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(scheduler_name, optimizer, lr, num_epochs, len(train_batch))
    
    loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.train()

    best_accuracy = 0
    best_params = None
    stop_increasing = 0

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_batch, 
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       leave=True)
        current_losses = []
        for batch in progress_bar:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss_history.append(loss.item())
            current_losses.append(loss.item())

            loss.backward()
            optimizer.step()  

            if scheduler_name == "OneCycleLR":
                scheduler.step()

        train_accuracy = calc_accuracy(model, train_batch, device=device)
        val_accuracy = calc_accuracy(model, val_batch, device=device)
        
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        avg_loss = sum(current_losses) / len(current_losses)

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_accuracy)

        if epoch % log_interval == 0: 
            print("Learning Rate:", scheduler.get_last_lr())
            print(f"Epoch {epoch+1} Loss: {avg_loss} Train Accuracy: {train_accuracy} \
            Validation Accuracy: {val_accuracy}")    

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = copy.deepcopy(model.state_dict())
            stop_increasing = 0
        else:
            stop_increasing += 1

        if stop_increasing == early_stop:
            print("Train Early Stop...")
            break
    
    # if best_params is not None:
    #     model.load_state_dict(best_params)
    
    return loss_history, train_accuracy_history, \
    val_accuracy_history, best_accuracy, best_params

def create_model(model_name, num_classes):
    if model_name == "swin":
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
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