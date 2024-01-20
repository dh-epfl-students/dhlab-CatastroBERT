import subprocess
"""
this scripts runs the training cycle for the different base models and was used to compare 
the performance of the different models
"""
def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def main():
    for seed in range(1, 6):
        print(f"Seed: {seed}")
        
        #run_command(["python3", "removing_years.py", str(seed), "camembert-base"])
        
       
        #run_command(["python3", "removing_years.py", str(seed), "bert-base-multilingual-cased"])

       

        #run_command(["python3", "removing_years.py", str(seed), "xlm-roberta-base","--batch_size", "16"])
        run_command(["python3", "removing_years.py", str(seed), "xlm-roberta-large", "--batch_size", "8"])
        

if __name__ == "__main__":
    main()
