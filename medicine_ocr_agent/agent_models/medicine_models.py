from uagents import Model 

class MedicineOCRRequest(Model):
    """ 
    A request model for fetching medicine information.
    Uses medicine name as the request parameter.
    """ 
    img_path: str = None

class MedicineOCRResponse(Model):
    """ 
    A response model for fetched medicine information.
    Contains the medicine name and a dictionary of medicine information.
    """ 
    medicines: list = None 
