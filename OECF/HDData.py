import exifread
import os
import rawpy
import numpy as np

class HDData:
    
    def __init__(self, folder_path, x, y, tag):
        self.__folder_path = folder_path
        self.__x = x
        self.__y = y
        self.__tag = tag
        self.__rad_pix_mean = 3
        
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
                raw_colors = raw.raw_colors

                # Sécurité
                if self.__y >= raw_data.shape[0] or self.__x >= raw_data.shape[1]:
                    continue
                
                # Récupération de la valeur des pixels autour de (y, x)
                y_min, y_max = max(0, self.__y-self.__rad_pix_mean), min(raw_data.shape[0], self.__y+self.__rad_pix_mean)
                x_min, x_max = max(0, self.__x-self.__rad_pix_mean), min(raw_data.shape[1], self.__x+self.__rad_pix_mean)
                
                # Extraction des blocs de données
                roi = raw_data[y_min:y_max, x_min:x_max].astype(np.float32) # Passage en float pour éviter les erreurs d'underflow
                roi_colors = raw_colors[y_min:y_max, x_min:x_max]

                # Récupération des niveaux de noir pour chaque canal (Sony ARW)
                black_levels = raw.black_level_per_channel

                # 4. Soustraction de l'offset noir AVANT la moyenne
                # On crée une matrice de la même taille que le ROI avec le bon offset pour chaque pixel
                roi_black_level = np.zeros_like(roi)
                for color_idx in range(4): # Pour chaque canal (0:R, 1:G1, 2:B, 3:G2)
                    roi_black_level[roi_colors == color_idx] = black_levels[color_idx]

                # (Optionnel mais critique pour l'OECF) : Soustraction du niveau de noir
                # Le capteur Sony ajoute un offset (ex: 512) pour éviter les valeurs négatives dues au bruit
                roi_linear = roi - roi_black_level

                # Sécurité : on remplace les valeurs négatives (dues au bruit dans les noirs) par 0
                roi_linear = np.clip(roi_linear, 0, None)

                # 5. Calcul de la moyenne sur le canal cliqué
                color_index = raw_colors[self.__y, self.__x]
                mask = (roi_colors == color_index)
                moyenne_pure = np.mean(roi_linear[mask])

            exposures.append(exposure_time)
            pixel_values.append(moyenne_pure)
            
        combined = sorted(zip(exposures, pixel_values), key=lambda x: x[0])

        exposures_sorted = [c[0] for c in combined]
        pixel_values_sorted = [c[1] for c in combined]

        return exposures_sorted, pixel_values_sorted
       
    def getListExpo(self):
        return self.__exposures
    
    def getListPixValues(self):
        return self.__pixel_values

    def getTag(self):
        return self.__tag
 