from uagents import Model 

class MedicineRequest(Model):
    """ 
    A request model for fetching medicine information.
    Uses medicine name as the request parameter.
    """ 
    medicine_names: list[str]
    is_save: bool = True

class MedicineResponse(Model):
    """ 
    A response model for fetched medicine information.
    Contains the medicine name and a dictionary of medicine information.
    """
    medicines_data: list[dict]
