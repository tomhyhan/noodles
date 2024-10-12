import os
from PIL import Image
import imagehash

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
    # return {h: paths for h, paths in hashes.items() if len(paths) > 1}

# Usage
# duplicates = find_duplicates('./Rigatoni')
# for hash, filepaths in duplicates.items():
#     print(f"Duplicate images (hash: {hash}):")
#     for path in filepaths:
#         print(f"  {path}")