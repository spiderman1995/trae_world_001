from typing import List
import datetime

# Representative list of ChiNext 50 stocks (as of roughly 2024)
# In a real system, this should be fetched from a database or API (e.g., AkShare/Tushare)
STATIC_CHINEXT50_LIST = [
    "300750", "300059", "300760", "300274", "300124", "300015", "300142", "300498", "300014", "300347",
    "300003", "300122", "300782", "300408", "300433", "300413", "300661", "300999", "300012", "300223",
    "300450", "300628", "300601", "300919", "300699", "300316", "300454", "300308", "300595", "300769",
    "300363", "300496", "300676", "300033", "300207", "300073", "300463", "300285", "300144", "300763",
    "300294", "300979", "300390", "300529", "300630", "300741", "300009", "300558", "300896", "300052"
]

def get_chinext50_constituents(date: str = None) -> List[str]:
    """
    Get ChiNext 50 constituents for a given date.
    
    Args:
        date (str): Date in 'YYYY-MM-DD' format.
        
    Returns:
        List[str]: List of stock codes (e.g., ['300750', '300059']).
    """
    # TODO: Implement dynamic historical query logic here.
    # For now, return the static list to ensure the pipeline works.
    return STATIC_CHINEXT50_LIST
