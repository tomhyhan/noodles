import os
import pandas as pd


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
    print(len(dataframe))
    dataframe.to_csv("./pasta_data.csv")



def main():
    pass


if __name__ == "__main__":
    main()

