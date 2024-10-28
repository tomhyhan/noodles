import os
import json
import imagehash
import requests
import pandas as pd
from time import sleep
from PIL import Image
from io import BytesIO
from PIL import Image

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

def image_search(search_term, api_key):
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    print(search_term, api_key)
    headers = {"Ocp-Apim-Subscription-Key" : api_key}

    totalEstimatedMatches = float('inf')
    offset = 0
    n_searches = 150
    while offset + n_searches <= totalEstimatedMatches:
        print(totalEstimatedMatches)
        params = {"q": search_term, 
                  "imageType": "photo", 
                  "count": n_searches,
                  "offset": offset
                }
    
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            totalEstimatedMatches = search_results["totalEstimatedMatches"]
            if not totalEstimatedMatches or totalEstimatedMatches == float("inf"):
                raise RuntimeError("totalEstimatedMatches should not be infinite")
            with open(f"{search_term}_{offset}.json", 'w') as json_file:
                json.dump(search_results, json_file)
        except Exception as err:
            print(err)
            exit()
        sleep(1)
        offset += 150
        n_searches = min(150, abs(totalEstimatedMatches - offset))
    return offset


def download_image(search_term, total_offset, option=None, extra_offset=0):
    os.makedirs(search_term, exist_ok=True)
    filename = f"{search_term} {option}" if option else search_term
    for offset in range(0, total_offset+1, 150):
        with open(f"{filename}_{offset}.json") as json_file:
            data = json.load(json_file)
            for i, image_data in enumerate(data["value"]):
                thumbnailUrl = image_data["thumbnailUrl"]
                if thumbnailUrl:
                    try:
                        response = requests.get(thumbnailUrl)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        rgb_image= image.convert("RGB")  
                        image_path = os.path.join(search_term, f"{search_term}_{offset+extra_offset+i}.jpg")
                        rgb_image.save(image_path, "JPEG")
                        
                    except Exception as e:
                        print(e)
                        print(f"Failed to process image from {thumbnailUrl}")
      
    
    
def find_duplicates(image_dir):
    hashes = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            with Image.open(filepath) as img:
                hash = str(imagehash.average_hash(img))
                if hash in hashes:
                    hashes[hash].append(filepath)
                else:
                    hashes[hash] = [filepath]
    
    for paths in hashes.values():
        if len(paths) > 1:
            for filepath in paths[1:]:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                except Exception as e:
                    print(e)

def create_csv(image_folder_path, class_encoder):
    classes = os.listdir(image_folder_path)
    
    labels = []
    for class_ in classes:
        class_folder = os.path.join(image_folder_path, class_)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                labels.append({
                    "img_path": img_path,
                    "label" : class_encoder[class_]
                })
    dataframe = pd.DataFrame(labels)
    dataframe = dataframe.sort_values(['label', 'img_path'])
    dataframe = dataframe.reset_index(drop=True)
    dataframe.to_csv("./pasta_data.csv")

def download_imgs(api_key):
    pasta_list = [
        "Spaghetti","Fettuccine","Penne","Rigatoni","Macaroni","Linguine","Farfalle","Tagliatelle","Fusilli","Orzo","Conchiglie","Bucatini","Orecchiette","Ravioli","Tortellini","Fregola"
    ]

    for pasta in pasta_list:
        first_offset= image_search(pasta, api_key)
        print("first_offset", first_offset)
        download_image(pasta, first_offset-150)
        second_offset= image_search(f"{pasta} noodles", api_key)
        print("second_offset", second_offset)
        download_image(pasta, second_offset-150, option="noodles", extra_offset=450)
        find_duplicates(pasta)