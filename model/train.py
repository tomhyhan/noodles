import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from torchvision.models import swin_s, Swin_S_Weights

def trainer(
    model: nn.Module,
    train_batch,
    val_batch,
    num_epochs,
    lr,
    batch_size,
    k_fold_id,
    weight_decay,
    early_stop=5,
    log_interval=1,
    device = "cpu"
):
    
    summary(model, input_size=(batch_size, 3, 224, 224))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.2, mode='max')
    
    loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.train()

    best_accuracy = 0
    best_model = None
    no_improvement = 0

    loss_history = []
    iterations = 0

    for epoch in range(num_epochs):
        losses = []
        progress_bar = tqdm(train_batch, 
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       leave=True)
        
        for batch in progress_bar:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()   
        
        train_accuracy = calc_accuracy(model, train_batch, device=device)
        val_accuracy = calc_accuracy(model, val_batch, device=device)
        avg_loss = sum(losses) // len(losses)
        
        scheduler.step(val_accuracy)

        if epoch % log_interval == 0: 
            print("Learning Rate:", scheduler.get_last_lr())
            print(f"Epoch {epoch+1} Loss: {avg_loss} Train Accuracy: {train_accuracy}")    

        if val_accuracy > best_accuracy:
            best_accuracy = best_accuracy
            best_model = model.
        else:
            no_improvement += 1

        if no_improvement == early_stop:
            break
    return loss_history, train_accuracy, val_accuracy, best_accuracy

def create_model(model_name, num_classes):
    if model_name == "swin":
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def calc_accuracy(model, batchs, device="cpu"):
    model.eval()
    
    corrects = 0
    total = 0
    
    for batch in batchs:
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        corrects += pred.argmax(dim=1).eq(y).sum().item()
        total += len(y)
        
    return corrects / total