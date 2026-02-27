import exifread
import os
import rawpy
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr, spsolve
import math

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
    
    def __init__(self, folder_path, height, width, bits):
        self.__folder_path = folder_path
        self.__height = height
        self.__width = width
        self.__ouverture = -0.1
        self.__iso = -1
        self.__focale = -0.1
        self.__bit_per_sample = bits
        
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
                num, den = str(exposure).split("/")
                return int(num) / int(den)
            else:
                return float(str(exposure))
            
    def gsolve(self, Z, B, l, w):
        # Z : Valeurs de pixels (i pixels, j images) - Array 2D
        # B : Log des temps d'exposition log(delta t) - Array 1D
        # l : Lambda (facteur de lissage)
        n = self.__bit_per_sample # 16384
        
        # Taille de la matrice A 
        Z1 = np.size(Z, 0) # num_samples
        Z2 = np.size(Z, 1) # nb_files
        A = lil_matrix((Z1 * Z2 + n + 1, n + Z1), dtype=np.float32)
        b = np.zeros(np.size(A, 0), dtype=np.float32)
        
        
        k = 0
        for i in range(Z1):
            for j in range(Z2):
                z = int(Z[i, j])
                wij = w[z]
                A[k, z] = wij
                A[k, n + i] = -wij
                b[k] = wij * B[j]
                k += 1
        
        # Fix the curve by setting its middle value to 0
        A[k, n//2] = 1
        k += 1

        # Include the smoothness equations
        for i in range(n-1):
            A[k, i]   =    l*w[i+1]
            A[k, i+1] = -2*l*w[i+1]
            A[k, i+2] =    l*w[i+1]
            k += 1
        
        print(np.shape(A))
        print(np.shape(b))
        
        # Solve the system using SVD
        A = A.tocsr()
        # pseudoA = np.linalg.pinv(A.toarray())
        x, istop, itn = lsqr(A, b)[:3]
        print (istop, itn)
        g = x[:n]
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

        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(".arw")
        ])
        
        # Variables de stockage de données
        exposures = []
        nb_files = len(files)
        num_samples = 50
        Z = np.zeros((num_samples, nb_files))
        
        # Chemin d'accés au fichier de stockage de z et des expositions
        Z_file_name = f"Z_{num_samples}_{nb_files}_.txt" 
        Z_file_path = os.path.join("data", Z_file_name)
        
        # On regarde si le fichier Z existe, dans ce cas on le charge
        if os.path.isfile(Z_file_path):
            Z, exposures = self.charger_matrice_Z(Z_file_path)
            
        # Il n'existe pas, on le crée à partir des images RAW
        else:
            
            # On séléctionne aléatoirement num_samples pixels dans l'image (même position pour toutes les images)
            y_coords = np.random.randint(0, self.__height, num_samples)
            x_coords = np.random.randint(0, self.__width, num_samples)
            
            for i, path in enumerate(files):
                
                afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))

                exposure_time = self.extract_parameters(path) # en secondes
                exposures.append(1/exposure_time) # en fréquence
                
                with rawpy.imread(path) as raw:
                    Z[:, i] = raw.raw_image[y_coords, x_coords]
            
            # Stockage de Z et les expositions
            self.stocker_matrice_Z(Z, exposures, Z_file_path)
        
        print(exposures)
        print(Z)
        
        # Dernier paramètrage des variables pour Debevec
        exposures = np.array(exposures, dtype=np.float32)
        w = [z if z <= 0.5*(self.__bit_per_sample - 1) else (self.__bit_per_sample - 1)-z for z in range(self.__bit_per_sample)]
        B = [np.log(e) for e in exposures]
        l = 10
        
        # Cacul de g et lE avec la méthode de Debevec
        print("Calcul de la courbe de réponse (Debevec)...")
        g, lE = self.gsolve(Z, B, l, w)
        
        pixel_values = np.arange(len(g)) 
        print(g)
        g = [np.log10(np.exp(i)) for i in g]
        plt.figure(figsize=(10, 6))

        # On trace g en X et les pixels en Y pour voir la courbe de réponse
        plt.plot(g, pixel_values, color='green', linewidth=2)

        plt.title("Courbe de Réponse du Capteur (OECF) - Méthode Debevec")
        plt.xlabel("Log Exposure (ln E * dt)")
        plt.ylabel("Valeur Numérique du Pixel (14-bit DN)")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.show()
        
        return g, lE, exposures
    
    def stocker_matrice_Z(self, Z, exposures, filename):
        try:
            # fmt="%.4f" permet de garder 4 décimales (suffisant pour du RAW)
            np.savetxt(filename, Z, fmt="%d", header=f"Matrice Z ({Z.shape[0]}x{Z.shape[1]})")
            np.savetxt(filename[:-4] + "_exposures.txt", exposures, fmt="%.10f", header=f"Exposition en fréquence par image")
            print(f"Matrice Z stockée avec succès dans {filename}")
        except Exception as e:
            print(f"Erreur lors du stockage : {e}")

    def charger_matrice_Z(self, filename):
        try:
            Z = np.loadtxt(filename)
            exposures = np.loadtxt(filename[:-4] + "_exposures.txt")
            print(f"Matrice Z chargée avec succès. Taille : {Z.shape}")
            return Z, exposures
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return None
       
    def getG(self):
        return self.__g

    def getTag(self):
        return self.__tag
 
    def getListExpo(self):
        return self.__exposures