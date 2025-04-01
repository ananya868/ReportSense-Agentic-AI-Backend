from uagents import Model


class MedicinePriceRequest(Model): 
    """ 
    A request model for fetching medicine price information.
    Uses medicine name as the request parameter.
    """ 
    medicine_names: list[str]

class MedicinePriceInfo(Model):
    """ 
    A model for storing medicine price information.
    Contains the medicine name, price, and URL.
    """ 
    name: str 
    price: str 
    quantity: str
    url: str

class MedicinePriceResponse(Model):
    """ 
    A response model for fetched medicine price information.
    Contains the medicine name and a dictionary of medicine price information.
    """ 
    medicine_price_info: Dict[str, List[MedicinePriceInfo]]