import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TransformToNull(BaseEstimator, TransformerMixin):
    """
    Transforma valores irregulares em nulos.
    
    Parâmetros:
    column_names: list of str
        Nomes das colunas no DataFrame a serem transformadas.
        
    A transformação converte:
      - Strings vazias em NaN.
      - Valores 'NM' ou 'Not Mentioned' em NaN.
      - Valores compostos apenas por caracteres especiais em NaN.
    """

    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        for col in self.column_names:
            # Converter strings vazias em NaN
            X_transformed[col] = X_transformed[col].replace('', np.nan)
            
            # Converter valores 'NM' ou 'Not Mentioned' para NaN
            X_transformed[col] = X_transformed[col].replace(['NM', 'Not Mentioned'], np.nan)
            
            # Se o valor conter apenas caracteres especiais, substitui por NaN
            special_mask = X_transformed[col].astype(str).str.fullmatch(r'[^a-zA-Z0-9]+')
            X_transformed.loc[special_mask, col] = np.nan
            
        return X_transformed

class StandardizeCategoryFormat(BaseEstimator, TransformerMixin):
    """
    Padroniza o formato dos dados categóricos removendo espaços desnecessários e uniformizando a capitalização.
    
    Parâmetros:
    column_names: list of str
        Nomes das colunas categóricas a serem padronizadas.
    """
    def __init__(self, column_names):
        self.column_names = column_names
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        for col in self.column_names:
            # Remove espaços extras e converte para Title Case
            X_transformed[col] = X_transformed[col].astype(str).str.strip().str.title()
        
        return X_transformed 
    


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Codifica os valores categóricos em números inteiros.
    
    Parâmetros:
    column_names: list of str
        Nomes das colunas categóricas a serem codificadas.
    
    Atributos:
    mapping_: dict
        Dicionário que mapeia os valores originais para inteiros para cada coluna.
    """
    def __init__(self, column_names):
        self.column_names = column_names
        self.mapping_ = {}
        
    def fit(self, X, y=None):
        for col in self.column_names:
            # Cria um mapeamento de cada valor único (excluindo nulos) para um inteiro
            unique_vals = sorted(X[col].dropna().unique())
            self.mapping_[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.column_names:
            mapping = self.mapping_.get(col, {})
            X_transformed[col] = X_transformed[col].map(mapping)
        return X_transformed
