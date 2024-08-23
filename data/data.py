import os
import json
from time import sleep

import dotenv
import requests
from io import BytesIO
from PIL import Image
import numpy as np


MAX_IMAGES = 1000
IMAGES_PER_REQUEST = 150

def image_search(search_term, api_key):
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key" : api_key}

    
    for offset in range(0, MAX_IMAGES, IMAGES_PER_REQUEST):
        
        params = {"q": search_term, 
                  "imageType": "photo", 
                  "count": IMAGES_PER_REQUEST,
                  "offset": offset
                }
    
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            with open(f"{search_term}_{offset}.json", 'w') as json_file:
                json.dump(search_results, json_file)
        except Exception as err:
            print(err)
            exit()
        sleep(1)

# thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]

def download_image(search_term):
    os.makedirs(search_term, exist_ok=True)
    for offset in range(0, MAX_IMAGES, IMAGES_PER_REQUEST):
        with open(f"{search_term}_{offset}.json") as json_file:
            data = json.load(json_file)
            for i, image_data in enumerate(data["value"]):
                thumbnailUrl = image_data["thumbnailUrl"]
                if thumbnailUrl:
                    try:
                        response = requests.get(thumbnailUrl)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        rgb_image= image.convert("RGB")  
                        resize_image = rgb_image.resize((224,224))
                        image_path = os.path.join(search_term, f"{search_term}_{offset+i}.jpg")
                        resize_image.save(image_path, "JPEG")
                        
                        # rgb_arr = np.array(resize_image)                
                        # print(image_data["thumbnailUrl"])
                        # print(rgb_arr)
                        # print(rgb_arr.shape)
                    except Exception as e:
                        print(e)
                        print(f"Failed to process image from {thumbnailUrl}")
    
def main():
    env = dotenv.dotenv_values()
    api_key = env.get("API_KEY")
    
    if not api_key:
        raise(RuntimeError("API KEY Not found!"))

    # image_search("Spaghetti", api_key)
    download_image("Spaghetti")
if __name__ == "__main__":
    main()
# 1. Spaghetti
# 2. Fettuccine
# 3. Penne
# 4. Rigatoni
# 5. Macaroni
# 6. Fusilli
# 7. Farfalle
# 8. Couscous
# 9. Fregola
# 10. Orzo
# 11. Conchiglie
# 12. Bucatini
# 13. Gemelli
