class Marker():
    
    def __init__(self, cordX: int, cordY: int, tag: str):
        self.__cordX = cordX
        self.__cordY = cordY
        self.__tag = tag
    
    def getCordX(self):
        return self.__cordX
    
    def getCordY(self):
        return self.__cordY
    
    def getTag(self):
        return self.__tag