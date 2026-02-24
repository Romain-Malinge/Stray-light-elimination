import exifread
import os
import rawpy
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

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
            
    def gsolve(self, Z, B, l, n = 16384):
        """
        Z : Valeurs de pixels (i pixels, j images) - Array 2D
        B : Log des temps d'exposition log(delta t) - Array 1D
        l : Lambda (facteur de lissage)
        w : Fonction de pondération (Array de taille n)
        """

        num_pixels = Z.shape[0]
        num_images = Z.shape[1]
        
        # Taille de la matrice A : (N*P + 1 + (N-2)) lignes , (N + P) colonnes
        A = lil_matrix((num_pixels * num_images + n + 1, n + num_pixels))
        b = np.zeros(A.shape[0])
        w = self.get_weighting_function()
        k = 0
        
        # Data-fitting
        for i in range(num_pixels):  
            for j in range(num_images):        
                z_val = int(Z[i, j])
                wij = w[z_val]  
                A[k, z_val] = wij
                A[k, n + i] = -wij
                b[k] = wij * B[j]
                k += 1
                 
        # Mettre le milieu de la courbe à 0
        A[k, n // 2] = 1
        k += 1
        
        # Équations de lissage
        for i in range(n - 2):
            A[k, i] = l * w[i + 1]
            A[k, i + 1] = -2 * l * w[i + 1]
            A[k, i + 2] = l * w[i + 1]
            k += 1
        
        # Résolution du système par SVD 
        A_csr = A.tocsr()
        x, istop, itn, normr = lsqr(A_csr, b)[:4]

        g = x[0:n]
        lE = x[n:]
    
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
        k = 0
        num_samples = 100
        Z = np.zeros((num_samples, nb_files))
        y_coords = np.random.randint(0, self.__height, num_samples)
        x_coords = np.random.randint(0, self.__width, num_samples)
        
        for i, path in enumerate(files):
            
            afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))

            exposure_time = self.extract_parameters(path)
            exposures.append(np.log10(exposure_time))
            
            with rawpy.imread(path) as raw:
                Z[:, k] = raw.raw_image[y_coords, x_coords]
                k += 1 
        
        g, lE = self.gsolve(Z, B=exposures, l=100)
        pixel_values = np.arange(len(g)) 
        

        plt.figure(figsize=(10, 6))

        # On trace g en X et les pixels en Y pour voir la courbe de réponse
        plt.plot(g, pixel_values, color='green', linewidth=2)

        plt.title("Courbe de Réponse du Capteur (OECF) - Méthode Debevec")
        plt.xlabel("Log Exposure (ln E * dt)")
        plt.ylabel("Valeur Numérique du Pixel (14-bit DN)")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.show()
        
        return g, lE, exposures
       
    def getG(self):
        return self.__g

    def getTag(self):
        return self.__tag
 
    def getListExpo(self):
        return self.__exposures