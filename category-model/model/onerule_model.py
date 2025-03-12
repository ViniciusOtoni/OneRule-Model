import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class OneRuleClassifier:
    """
    Implementação simples do algoritmo OneRule.
    Para cada feature, gera uma regra que, para cada valor, prevê a classe majoritária.
    Em seguida, seleciona a feature que minimiza o erro total de classificação.
    """
    def __init__(self):
        self.best_feature = None
        self.rule = {}
        self.error = None
        self.default_class = None  # Valor padrão para casos não mapeados

    def fit(self, X, y):
        best_error = float('inf')
        best_feature = None
        best_rule = {}
        # Define o valor padrão global (classe majoritária) para o caso de valores não vistos
        self.default_class = y.mode()[0]
        
        # Itera sobre cada feature para criar a regra
        for feature in X.columns:
            rule = {}
            error = 0
            # Para cada valor único da feature, define a classe majoritária
            for value in X[feature].unique():
                mask = X[feature] == value
                majority_class = y[mask].mode()[0]
                rule[value] = majority_class
                error += (y[mask] != majority_class).sum()
            # Seleciona a feature com menor erro total
            if error < best_error:
                best_error = error
                best_feature = feature
                best_rule = rule
                
        self.best_feature = best_feature
        self.rule = best_rule
        self.error = best_error

    def predict(self, X):
        predictions = []
        # Utiliza a regra da melhor feature para cada exemplo
        for _, row in X.iterrows():
            value = row[self.best_feature]
            # Se o valor não estiver mapeado, utiliza o valor padrão (classe majoritária)
            predictions.append(self.rule.get(value, self.default_class))
        return predictions


class OneRuleCreditApprovalModel:
    """
    Modelo para prever a aprovação de crédito utilizando o algoritmo OneRule.
    
    Dataset esperado:
      - antecedentes_criminais
      - profissao
      - carga_horaria
      - estado_civil
      - renda_familiar
      - possui_imovel
      - tempo_emprego
      - garantias
      - faixa_etaria
      - tipo_operacao
      - score_credito
      
    O target 'aprovacao_credito' é derivado de 'score_credito':
      - Se score_credito for codificado como 0 ("Alto"), então aprovação é 1.
      - Caso contrário, aprovação é 0.
      
    Todas as demais colunas são utilizadas como features.
    """
    def __init__(self, cleaned_data_path):
        self.cleaned_data_path = cleaned_data_path
        self.model = OneRuleClassifier()

    def load_data(self):
        self.df = pd.read_csv(self.cleaned_data_path, header=0)

    def prepare_data(self):
        if 'score_credito' not in self.df.columns:
            raise ValueError("A coluna 'score_credito' não foi encontrada no dataset.")
        
        # Deriva o target: crédito aprovado (1) se score_credito for 0 ("Alto"), 0 caso contrário.
        self.df['aprovacao_credito'] = self.df['score_credito'].apply(lambda x: 1 if x == 0 else 0)
        
        # Define as features removendo score_credito e o target derivado
        self.X = self.df.drop(['score_credito', 'aprovacao_credito'], axis=1)
        self.y = self.df['aprovacao_credito']
        
        print("Dados preparados. Features:", self.X.columns.tolist())

    def train_model(self):
        # Divide os dados em treinamento (80%) e teste (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    


        print(self.y.value_counts())
        print("Acurácia do modelo:", acc)
        print("Relatório de Classificação:")
        print(classification_report(y_test, y_pred, zero_division=0))

    def run(self):
        """
        Executa o pipeline completo: carrega os dados, prepara, treina e avalia o modelo.
        """
        self.load_data()
        self.prepare_data()
        self.train_model()


