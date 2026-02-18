import exifread
import os
import rawpy
import numpy as np
import sys

def afficher_progression(actuel, total, nom_fichier=""):
        """Affiche une barre de progression simple dans le terminal."""
        largeur_barre = 40
        progression = actuel / total
        nb_carres = int(largeur_barre * progression)
        
        # Création de la ligne : [████░░░░] 50% - fichier.arw
        barre = "█" * nb_carres + "░" * (largeur_barre - nb_carres)
        pourcentage = int(progression * 100)
        
        # \r ramène le curseur au début de la ligne
        sys.stdout.write(f"\r|{barre}| {pourcentage}% - Analyse : {nom_fichier}")
        sys.stdout.flush() # Force l'affichage immédiat

class HDData:
    
    def __init__(self, folder_path, x, y, tag):
        self.__folder_path = folder_path
        self.__x = x
        self.__y = y
        self.__tag = tag
        self.__rad_pix_mean = 0
        
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
        
        nb_files = len(files)

        for i, path in enumerate(files):
            
            afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))

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
                
                # Extraction des blocs de données
                roi = raw_data[self.__y, self.__x] # Passage en float pour éviter les erreurs d'underflow

            exposures.append(exposure_time) # coordonnée x du pixel
            pixel_values.append(roi) # coordonnée y du pixel
            
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
 