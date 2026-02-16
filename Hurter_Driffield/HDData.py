import exifread
import math
import os
import rawpy

class HDData:
    
    def __init__(self, folder_path, x, y, tag):
        self.__folder_path = folder_path
        self.__x = x
        self.__y = y
        self.__tag = tag
        
        self.__exposures, self.__pixel_values = self.extract_hd_data()
    
    def extract_exposure_time(self, path):
        """
        Extrait le temps de pose EXIF (en secondes)
        """
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="EXIF ExposureTime")

            exposure = tags.get("EXIF ExposureTime")
            if exposure is None:
                return None

            # ex: 1/8000
            if "/" in str(exposure):
                num, den = map(float, str(exposure).split("/"))
                return num / den
            else:
                return float(str(exposure))
            
    def extract_hd_data(self):
        """
        Récupère les données nécessaires pour tracer la courbe H&D
        pour un pixel (x,y).
        
        Retourne :
            exposures_log : liste log10(temps de pose)
            pixel_values  : liste valeurs RAW correspondantes
        """

        exposures = []
        pixel_values = []
        
        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(".arw")
        ])

        for path in files:

            exposure_time = self.extract_exposure_time(path)
            if exposure_time is None:
                continue

            with rawpy.imread(path) as raw:
                # Accès à la matrice brute (Bayer) non développée
                # C'est un tableau numpy en 16 bits (souvent 12 ou 14 bits réels)
                raw_data = raw.raw_image

                # Sécurité
                if self.__y >= raw_data.shape[0] or self.__x >= raw_data.shape[1]:
                    continue
                
                # Récupération de la valeur au pixel (y, x)
                value = raw_data[self.__y, self.__x]
                
                # (Optionnel mais critique pour l'OECF) : Soustraction du niveau de noir
                # Le capteur Sony ajoute un offset (ex: 512) pour éviter les valeurs négatives dues au bruit
                black_level = raw.black_level_per_channel[raw.raw_color_indices[self.__y, self.__x]]
                linear_value = max(0, value - black_level)

            exposures.append(exposure_time)
            pixel_values.append(linear_value)

        # Convertir en log10
        exposures_log = [math.log10(t) for t in exposures]

        # Trier par exposition croissante
        combined = sorted(zip(exposures_log, pixel_values), key=lambda x: x[0])

        exposures_log_sorted = [c[0] for c in combined]
        pixel_values_sorted = [c[1] for c in combined]

        return exposures_log_sorted, pixel_values_sorted
       
    def getListExpo(self):
        return self.__exposures
    
    def getListPixValues(self):
        return self.__pixel_values

    def getTag(self):
        return self.__tag
 