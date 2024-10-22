import dotenv

from data.data import image_search, download_image
from duplicate import find_duplicates

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

def main():
    env = dotenv.dotenv_values()
    api_key = env.get("API_KEY")
    
    if not api_key:
        raise(RuntimeError("API KEY Not found!"))

    # pasta_list = ["Penne" , "Rigatoni" , "Macaroni", "Fusilli", "Farfalle", "Couscous", "Fregola", "Orzo", "Conchiglie", "Bucatini", "Gemelli"]
    pasta_list = ["Linguine" , "Tagliatelle" , "Orecchiette", "Ravioli", "Tortellini"]

    for pasta in pasta_list:
        first_offset= image_search(pasta, api_key)
        print("first_offset", first_offset)
        download_image(pasta, first_offset-150)
        second_offset= image_search(f"{pasta} noodles", api_key)
        print("second_offset", second_offset)
        download_image(pasta, second_offset-150, option="noodles", extra_offset=450)
        find_duplicates(pasta)
    
    
if __name__ == "__main__":
    main()