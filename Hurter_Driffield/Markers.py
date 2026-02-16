class Marker():
    
    def __init__(self, cordX: int, cordY: int, id: int):
        self.__cordX = cordX
        self.__cordY = cordY
        self.__id = id
    
    def getCordX(self):
        return self.__cordX
    
    def getCordY(self):
        return self.__cordY
    
    def getID(self):
        return self.__id