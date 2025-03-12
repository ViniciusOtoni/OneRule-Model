import requests

class GenerateEngine:

    def __init__(self, url, fields, params):

        self.url = url
        self.fields = fields
        self.params = params

    def generateData(self):
        response = requests.post(self.url, json=self.fields, params=self.params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"erro na requisição: {response.status_code}")
            print(f"motivo do erro: {response.text}") 

    def getMockarooDataTypes(self):
        
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"erro na requisição: {response.status_code}")
            print(f"motivo do erro: {response.text}") 
