import exifread
import os
import rawpy
import numpy as np
import sys
import cv2

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
    
    def __init__(self, folder_path, x, y, tag, height, width):
        self.__folder_path = folder_path
        self.__x = x
        self.__y = y
        self.__tag = tag
        self.__height = height
        self.__width = width
        self.__ouverture = -0.1
        self.__iso = -1
        self.__focale = -0.1
        
        self.__g, self.__lE, self.__exposures = self.extract_hd_data()
    
    def extract_parameters(self, path):
        """
        Extrait le temps de pose EXIF (en secondes)
        """
        with open(path, "rb") as f:
            tags = exifread.process_file(f)
            
            focale = tags.get('EXIF FocalLength')
            if focale is not None:
                if self.__focale == -0.1:
                    self.__focale = float(focale.values[0])
                elif self.__focale != float(focale.values[0]):
                    print(f"Erreur, lors de votre prise de photo vous avez changé de focale à la photo {path}")
                    return False
            
            iso = tags.get('EXIF ISOSpeedRatings')
            if iso is not None:
                if self.__iso == -1:
                    self.__iso = int(iso.values[0])
                elif self.__iso != int(iso.values[0]) :
                    print(f"Erreur, lors de votre prise de photo vous avez changé d'iso à la photo {path}")
                    return False
            
            ouverture = tags.get('EXIF FNumber')
            if ouverture is not None:
                if self.__ouverture == -0.1:
                    self.__ouverture = float(ouverture.values[0])
                elif self.__ouverture != float(ouverture.values[0]):
                    print(f"Erreur, lors de votre prise de photo vous avez changé d'ouverture à la photo {path}")
                    return False
                
            exposure = tags.get("EXIF ExposureTime")
            if exposure is None:
                return None

            # ex: 1/8000
            if "/" in str(exposure):
                num, den = map(float, str(exposure).split("/"))
                return num / den
            else:
                return float(str(exposure))
            
    def get_weighting_function(self, n=16384):
        """
        n : nombre de niveaux (256 pour 8-bit, 16384 pour 14-bit)
        """
        z_min = 0
        z_max = n - 1
        z_mid = (z_min + z_max) / 2
        
        # Création du tableau de poids
        w = np.array([z - z_min if z <= z_mid else z_max - z for z in range(n)])
        
        # On s'assure que les poids sont de type float pour les calculs
        return w.astype(np.float32)
            
    def gsolve(self, B, l):
        """
        Z : Valeurs de pixels (i pixels, j images) - Array 2D
        B : Log des temps d'exposition log(delta t) - Array 1D
        l : Lambda (facteur de lissage)
        w : Fonction de pondération (Array de taille n)
        """
        n = 256 # Passer à 16384 pour du RAW 14-bits
        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(".arw")
        ])
        num_pixels = self.__height * self.__width
        num_images = len(files)
        
        # Taille de la matrice A : (N*P + 1 + (N-2)) lignes , (N + P) colonnes
        # N = niveaux de gris, P = nombre de pixels échantillonnés
        A = np.zeros((num_pixels * num_images + n + 1, n + num_pixels))
        b = np.zeros((A.shape[0], 1))
        w = self.get_weighting_function()
        
        k = 0
        
        ## 1. Équations de correspondance des données (Data-fitting)
        for j, path in enumerate(files):
            with rawpy.imread(path) as raw:
                for i in range(num_pixels):  
                        
                    z_val = raw.raw_image[i % self.__width, i // self.__height]
                    wij = w[z_val]
                    A[k, z_val] = wij
                    A[k, n + i] = -wij
                    b[k, 0] = wij * B[j]
                    k += 1
                
        ## 2. Fixer le milieu de la courbe à 0 (Ancrage de l'exposition)
        # Dans le papier, g(128) = 0. En Python (index 0), c'est 127
        A[k, 128] = 1
        k += 1
        
        ## 3. Équations de lissage (Smoothness)
        # Assure que la courbe est "belle" (dérivée seconde faible)
        for i in range(n - 2):
            A[k, i] = l * w[i + 1]
            A[k, i + 1] = -2 * l * w[i + 1]
            A[k, i + 2] = l * w[i + 1]
            k += 1
            
        ## 4. Résolution du système par moindres carrés (SVD interne)
        # En Matlab 'A\b', en Python 'np.linalg.lstsq'
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        g = x[0:n].flatten()
        lE = x[n:].flatten()
    
        return g, lE
            
    def extract_hd_data(self):
        """
        Récupère les données nécessaires pour tracer la courbe H&D
        pour un pixel (x,y).
        
        Retourne :
            exposures_log : liste log10(temps de pose)
            pixel_values  : liste valeurs RAW correspondantes
        """

        exposures = []
        
        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(".arw")
        ])
        
        nb_files = len(files)

        for i, path in enumerate(files):
            
            afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))

            exposure_time = self.extract_parameters(path)
            exposures.append(np.log10(exposure_time))
            
        g, lE = self.gsolve(B=exposures, l=100)
        
        return g, lE, exposures
       
    def getG(self):
        return self.__g
    
    def getLE(self):
        return self.__lE

    def getTag(self):
        return self.__tag
 
    def getListExpo(self):
        return self.__exposures