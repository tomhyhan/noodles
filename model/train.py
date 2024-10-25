import torch
from torch import nn
from torch import optim
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau

def trainer(
    model: nn.Module,
    train_batch,
    val_batch,
    num_epochs,
    lr,
    weight_decay,
    device = "cpu"
):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.2, mode='max')
    
    loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.train()
    
    loss_history = []
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
        avg_loss = sum(losses) // len(losses)
        
        scheduler.step(train_accuracy)
        print("Learning Rate:", scheduler.get_last_lr())
        print(f"Epoch {epoch+1} Loss: {avg_loss} Train Accuracy: {train_accuracy}")    
    

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