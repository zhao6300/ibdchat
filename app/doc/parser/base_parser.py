
class BaseParser:
    def parse(self)-> list
        """
        Parses the document and returns a list of parsed items.
        
        Returns:
            list: A list of parsed items.
        """
        raise NotImplementedError("Subclasses must implement this method.")
