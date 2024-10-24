import os
import dotenv
import pandas as pd
from sklearn.model_selection import train_test_split

from model.data import image_search, download_image, create_csv, find_duplicates

# 1. Spaghetti v
# 2. Fettuccine v
# 3. Penne v
# 4. Rigatoni v
# 5. Macaroni v
# 6. Linguine o
# 7. Farfalle x
# 8. Tagliatelle x
# 9. Fusilli x
# 10. Orzo x
# 11. Conchiglie x
# 12. Bucatini t
# 13. Orecchiette t
# 14. Ravioli t
# 15. Tortellini t
# 16. Fregola
CLASS_ENCODER = {
        "Spaghetti" : 0,
        "Fettuccine": 1,
        "Penne": 2,
        "Rigatoni": 3,
        "Macaroni": 4,
        "Linguine": 5,
        "Farfalle": 6,
        "Tagliatelle": 7,
        "Fusilli": 8,
        "Orzo": 9,
        "Conchiglie": 10,
        "Bucatini": 11,
        "Orecchiette": 12,
        "Ravioli": 13,
        "Tortellini": 14,
        "Fregola": 15
    }
SEED = 42

def main():
    env = dotenv.dotenv_values()
    api_key = env.get("API_KEY")
    
    if not api_key:
        raise(RuntimeError("API KEY Not found!"))
    image_folder_path = os.path.join(os.getcwd(), "Images") 
    # download_image(api_key)
    # create_csv(image_folder_path, CLASS_ENCODER)

    data = pd.read_csv("./pasta_data.csv")
    image_paths, labels = data["img_path"], data["label"]
    
    X, test_data, y, test_label = train_test_split(image_paths.values, labels.values, train_size=0.9, random_state=SEED, shuffle=True, stratify=labels)
    
    print(test_data[0])
    print(len(X), len(test_data))
    print(type(X))
    print(y[0], test_label[0])
    
    
if __name__ == "__main__":
    main()