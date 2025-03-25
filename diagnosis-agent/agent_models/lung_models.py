from uagents import Model

class LungRequest(Model):
    file_path: str

class LungResponse(Model):
    cancer_prediction: str
