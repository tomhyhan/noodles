# Image classification Model for pasta noodles 

## The Model classifies the following categories
1. Spaghetti
2. Fettuccine
3. Penne
4. Rigatoni
5. Macaroni
6. Linguine
7. Farfalle
8. Tagliatelle
9. Fusilli
10. Orzo
11. Conchiglie
12. Bucatini
13. Orecchiette
14. Ravioli
15. Tortellini
16. Fregola

## Process

1. Create dataframe for images and label ✔
2. Save dataframe into CSV file [image_file_path, label] ✔
3. Load dataframe and split the data into train/test ✔
4. Seperatly calculate the mean and std for train_set ✔
5. Further split train set into train/val 
6. Define Tranform for the image
7. Create Custom Dataset that takes int Dataframe and Transform
8. Load Train and Validation set into DataLoader (K-fold)
9. Define Trainer
10. Train the model on:
    - Swin Transformer
    - MaxVIT
    - RegNet
    - Efficient Net

# Optional
    1. Organize structure
    2. create config file

# K-fold
    - Iterate through K-fold, and find out the model that has the best accuracy

# Summary
    - use torch summary