from uagents import Model


class MedicinePriceRequest(Model): 
    """ 
    A request model for fetching medicine price information.
    Uses medicine name as the request parameter.
    """ 
    medicine_name: str
    is_save: bool = True

class MedicinePriceResponse(Model):
    """ 
    A response model for fetched medicine price information.
    Contains the medicine name and a dictionary of medicine price information.
    """ 
    medicine_name: str
    medicine_price_info: dict