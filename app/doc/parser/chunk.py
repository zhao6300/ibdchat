

class ChunkBase:
    def encode(self) -> list:
        """
        Encode the chunk to a string representation.
        """
        raise NotImplementedError("Subclasses should implement this method.")
