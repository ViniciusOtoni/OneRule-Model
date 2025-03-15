import sys
import os
import json
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from generate.main import GenerateEngine
from pipelineProcess.pipeline import Pipe
from model.onerule_model import CreditApprovalModel
from pipelineProcess.ETL import EncodeCategorical

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



def decode_categorical(df, encoder):
    df_decoded = df.copy()
    for col in encoder.column_names:
        if col in df_decoded.columns and col in encoder.mapping_:
            # Cria o mapeamento inverso: {valor_codificado: valor_original}
            inv_mapping = {v: k for k, v in encoder.mapping_[col].items()}
            df_decoded[col] = df_decoded[col].map(inv_mapping)
    return df_decoded


def oneRuleModel() -> None:
    cleaned_data_path = "../data/cleaned/dataset.csv"
    
    
    df = pd.read_csv(cleaned_data_path, header=0)
    
    
    df['aprovacao_credito'] = df['score_credito'].apply(lambda x: 1 if x == 0 else 0)
    
    
    credit_model = CreditApprovalModel(
        data_path=cleaned_data_path,
        verbose=True  # Altere para False se não quiser logs detalhados
    )
    credit_model.load_and_prepare_data()  # Carrega e prepara os dados internamente
    credit_model.train()  # Treina o modelo
    
    
    X = df.drop(['score_credito', 'aprovacao_credito'], axis=1)
    predictions = credit_model.model.predict(X)
    df['aprovacao_credito_prevista'] = predictions
    
    
    categorical_cols = ['antecedentes_criminais', 'profissao', 'carga_horaria',
                       'estado_civil', 'renda_familiar', 'possui_imovel',
                       'tempo_emprego', 'garantias', 'faixa_etaria', 'tipo_operacao', 'score_credito']
    
    encoder = EncodeCategorical(categorical_cols)
    df_original = pd.read_csv("../data/raw/dataset.csv", header=0)
    encoder.fit(df_original)
    df_final = decode_categorical(df, encoder)
    
    
    output_dir = os.path.join("..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")
    df_final.to_csv(output_path, index=False)

def main():
    generate_data()
    pipeline()
    
    #Execução do modelo:
    oneRuleModel()




if __name__ == "__main__":
    main()
