from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class OneRuleClassifier:
    """
    Implementação aprimorada do algoritmo OneRule com:
    - Tratamento de valores ausentes/não vistos
    - Verificação de distribuição de classes
    - Logs detalhados para debug
    """
    def __init__(self, verbose=False):
        self.best_feature = None
        self.feature_rules = {}
        self.default_class = None
        self.verbose = verbose
        self.class_distribution = None

    def _create_rule_for_feature(self, X_feature, y):
        rule = {}
        error = 0
        total = 0
        
        # Agrupa por valores da feature e encontra a classe majoritária
        value_counts = X_feature.value_counts()
        
        for value, count in value_counts.items():
            subset = y[X_feature == value]
            if subset.empty:
                continue
                
            majority_class = subset.mode()[0]
            class_counts = subset.value_counts()
            rule[value] = {
                'class': majority_class,
                'confidence': class_counts[majority_class] / len(subset)
            }
            error += len(subset) - class_counts[majority_class]
            total += len(subset)

        # Calcula erro total e trata divisão por zero
        error_rate = error / total if total > 0 else 1.0
        return rule, error_rate

    def fit(self, X, y):
        if len(y.unique()) == 1:
            raise ValueError("Todos os exemplos pertencem à mesma classe")
            
        self.class_distribution = y.value_counts(normalize=True)
        self.default_class = y.mode()[0]
        
        best_error = float('inf')
        best_feature = None
        best_rule = {}

        for feature in X.columns:
            feature_data = X[feature]
            rule, error_rate = self._create_rule_for_feature(feature_data, y)
            
            if self.verbose:
                print(f"Feature: {feature}")
                print(f"Error rate: {error_rate:.4f}")
                print(f"Rule: {rule}\n")
            
            if error_rate < best_error:
                best_error = error_rate
                best_feature = feature
                best_rule = rule

        self.best_feature = best_feature
        self.feature_rules = best_rule
        
        if self.verbose:
            print(f"Melhor feature: {best_feature}")
            print(f"Erro: {best_error:.4f}")
            print(f"Regra final: {best_rule}")

    def predict(self, X):
        if self.best_feature is None:
            raise ValueError("Modelo não foi treinado")
            
        predictions = []
        feature_values = X[self.best_feature]
        
        for value in feature_values:
            if value in self.feature_rules:
                predictions.append(self.feature_rules[value]['class'])
            else:
                predictions.append(self.default_class)
                
        return pd.Series(predictions, index=X.index)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"Acurácia: {acc:.4f}")
        print("Relatório de Classificação:")
        print(classification_report(y, y_pred, zero_division=0))

class CreditApprovalModel:
    """
    Modelo aprimorado para aprovação de crédito com:
    - Validação de dados
    - Balanceamento de classes
    - Análise exploratória integrada
    """
    def __init__(self, data_path, test_size=0.2, random_state=42, verbose=False):
        self.data_path = data_path
        self.model = OneRuleClassifier(verbose=verbose)
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def load_and_prepare_data(self):
        # Carregar e validar dados
        self.data = pd.read_csv(self.data_path)
        
        if 'score_credito' not in self.data.columns:
            raise ValueError("Coluna 'score_credito' não encontrada")
            
        # Criar target
        self.data['aprovacao_credito'] = self.data['score_credito'].map({0: 1, 1: 0, 2: 0})
        
        # Verificar balanceamento
        class_dist = self.data['aprovacao_credito'].value_counts()
        print("Distribuição de Classes:")
        print(class_dist)
        
        if abs(class_dist[0] - class_dist[1]) > 0.7 * len(self.data):
            print("\nAviso: Desbalanceamento significativo de classes detectado!")

        # Separar features e target
        X = self.data.drop(['score_credito', 'aprovacao_credito'], axis=1)
        y = self.data['aprovacao_credito']
        
        # Divisão treino-teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
        print("\nPerformance no Treino:")
        self.model.evaluate(self.X_train, self.y_train)
        
        print("\nPerformance no Teste:")
        self.model.evaluate(self.X_test, self.y_test)

    def get_feature_importance(self):
        if self.model.best_feature is None:
            return None
            
        return {
            'best_feature': self.model.best_feature,
            'rules': self.model.feature_rules,
            'default_class': self.model.default_class
        }

    def run_pipeline(self):
        self.load_and_prepare_data()
        self.train()
        return self.get_feature_importance()