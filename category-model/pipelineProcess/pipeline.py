from sklearn.pipeline import Pipeline
import pandas as pd
import sys

sys.path.insert(0, "../pipelineProcess")
import ETL

class Pipe:

    def __init__(self, df):
        self.df = df

    def create_pipeline(self):
       categorical_columns = [
                                "antecedentes_criminais", "profissao", "carga_horaria",
                                "estado_civil", "renda_familiar", "possui_imovel", "tempo_emprego",
                                "garantias", "faixa_etaria", "tipo_operacao", "score_credito" ]
       
       str_pipe = Pipeline(steps=[
            ('transform_to_null', ETL.TransformToNull(categorical_columns)),
            ('standardize_format', ETL.StandardizeCategoryFormat(categorical_columns)),
            ('encode_categorical', ETL.EncodeCategorical(categorical_columns))
       ])
       
       return str_pipe
    
    def fitPipeline(self, pipeline, df):
        return pipeline.fit(df)

    def transformPipeline(self, model, df):
        return model.transform(df)


    def execute(self):
        pipeline = self.create_pipeline()
        model = self.fitPipeline(pipeline, self.df)
        df_transformed = self.transformPipeline(model, self.df)
        return df_transformed