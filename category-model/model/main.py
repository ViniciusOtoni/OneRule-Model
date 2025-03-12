import sys
import os
import json
import datetime
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from generate.main import GenerateEngine
from pipelineProcess.pipeline import Pipe
from model.onerule_model import OneRuleCreditApprovalModel

def create_csv_file(data) -> None:
    output_dir = os.path.join(parent_dir, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(data)
    file_name = f"dataset.csv"
    file_path = os.path.join(output_dir, file_name)
    
    df.to_csv(file_path, index=False)
    print("Arquivo CSV criado com sucesso!")

def generate_data() -> None:
    mockaroo_key = "ec5a90c0"
    api_url = "https://api.mockaroo.com/api/generate.json"
    params = {
        "key": mockaroo_key,
        "count": 1000
    }

    with open(os.path.join(current_dir, 'fields.json'), 'r') as file:
        fields = json.load(file)

    mock = GenerateEngine(api_url, fields, params)
    data = mock.generateData()

    if data:
        create_csv_file(data)
    else:
        print("Nenhum dado foi gerado.")

def pipeline() -> None:
    df = pd.read_csv("../data/raw/dataset.csv", header=0)

    pipe = Pipe(df)
    df_new = pipe.execute()

    output_dir = "../data/cleaned"
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"dataset.csv"
    file_path = os.path.join(output_dir, file_name)

    df_new.to_csv(file_path, index=False)



def oneRuleModel() -> None:
    cleaned_data_path = "../data/cleaned/dataset.csv"
    model = OneRuleCreditApprovalModel(cleaned_data_path)
    model.run()



def main():
    generate_data()
    pipeline()
    
    #Execução do modelo:
    oneRuleModel()




if __name__ == "__main__":
    main()
