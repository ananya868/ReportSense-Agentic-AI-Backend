from uagents import Model 

class MedicineRequest(Model):
    """ 
    A request model for fetching medicine information.
    Uses medicine name as the request parameter.
    """ 
    medicine_name: str
    is_save: bool = True

class MedicineResponse(Model):
    """ 
    A response model for fetched medicine information.
    Contains the medicine name and a dictionary of medicine information.
    """ 
    medicine_name: str
    medicine_info: dict 
