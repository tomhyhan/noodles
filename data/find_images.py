import os
import json
from time import sleep

import requests
from io import BytesIO
from PIL import Image

def image_search(search_term, api_key):
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    print(search_term, api_key)
    headers = {"Ocp-Apim-Subscription-Key" : api_key}

    totalEstimatedMatches = float('inf')
    offset = 0
    n_searches = 150
    # data["totalEstimatedMatches"]
    while offset + n_searches <= totalEstimatedMatches:
    # for offset in range(0, MAX_IMAGES, IMAGES_PER_REQUEST):
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

# thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]

def download_image(search_term, total_offset, option=None, extra_offset=0):
    os.makedirs(search_term, exist_ok=True)
    filename = f"{search_term} {option}" if option else search_term
    # total_offset = total_offset if total_offset % 150 == 0 else total_offset + 1
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
                        
                        # rgb_arr = np.array(resize_image)                
                        # print(image_data["thumbnailUrl"])
                        # print(rgb_arr)
                        # print(rgb_arr.shape)
                    except Exception as e:
                        print(e)
                        print(f"Failed to process image from {thumbnailUrl}")
      
    
def main():
    pass

if __name__ == "__main__":
    main()

