import exifread
import os
import rawpy
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import ResponseCalculator as RC
import imageio

SUPPORTED_EXTENSIONS = (".arw", ".jpg", ".jpeg", ".png", ".nef")

def afficher_progression(actuel, total, nom_fichier=""):
    largeur_barre = 40
    progression = actuel / total
    nb_carres = int(largeur_barre * progression)
    barre = "█" * nb_carres + "░" * (largeur_barre - nb_carres)
    pourcentage = int(progression * 100)
    sys.stdout.write(f"\r|{barre}| {pourcentage}% - Analyse : {nom_fichier}")
    sys.stdout.flush()

class HDData:
    
    def __init__(self, folder_path, height, width, bits, use_matlab=False, bases = {"Spline": True} ):
        self.__folder_path = folder_path
        self.__height = height
        self.__width = width
        self.__bit_per_sample = bits
        self.use_matlab = use_matlab
        self.bases = bases
        
        # Initial values for consistency checks
        self.__focale = -0.1
        self.__iso = -1
        self.__ouverture = -0.1
        
        # Load manual exposure table if it exists
        self.__manual_exposures = self._load_exposure_table()
        
        self.extract_hd_data()
    
    def _load_exposure_table(self):
        """Parses the exposure.txt file provided by the user."""
        table = {}
        # Looking for 'exposure.txt' in the folder
        file_path = os.path.join(self.__folder_path, "exposure.txt")
        if os.path.exists(file_path):
            print(f"Fichier d'exposition trouvé : {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        try:
                            # We take the middle column (shutter speed in seconds)
                            expo_val = float(parts[1])
                            table[filename] = expo_val
                        except ValueError:
                            continue
        return table

    def extract_parameters(self, path):
        filename = os.path.basename(path)
        
        # Priority 1: Check the manual exposure table
        if filename in self.__manual_exposures:
            return self.__manual_exposures[filename]
            
        # Priority 2: Try EXIF
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f)
                
                # On récupère l'exposition
                exposure = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
                if exposure:
                    if "/" in str(exposure):
                        num, den = str(exposure).split("/")
                        return int(num) / int(den)
                    return float(str(exposure))
                
                # On vérifie la consistance des autres paramètres
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
                
        except:
            pass
            
        print(f"\nAttention: Pas d'exposition trouvée pour {filename}. Valeur par défaut 1.0 utilisée.")
        return 1.0 
                    
    def extract_hd_data(self):
        
        # Récupération de la liste des images dans le répertoire
        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(SUPPORTED_EXTENSIONS) and f.lower() != "exposure.txt"
        ])
        
        # Paramètres de base
        nb_files = len(files)
        num_samples = 100
        
        # Variables de stockage de données
        exposures = np.ndarray(shape=(nb_files,), dtype=np.float32)
        Z = np.ndarray(shape=(num_samples, nb_files), dtype=np.uint16)
        
        # Chemin d'accés au fichier de stockage de z et des expositions
        file_name = f"{os.path.basename(self.__folder_path)}_{num_samples}samples_{nb_files}files"
        file_path = os.path.join("data", file_name)
        
        if os.path.isfile(file_path + "_Z.txt") and os.path.isfile(file_path + "_exposures.txt"):
            Z, exposures = self.charger_matrices(file_path)
        else:
            y_coords = np.random.randint(0, self.__height, num_samples)
            x_coords = np.random.randint(0, self.__width, num_samples)
            
            for i, path in enumerate(files):
                afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))
                exposure_time = self.extract_parameters(path)
                exposures[i] = exposure_time
                
                if path.lower().endswith((".arw", ".nef")):
                    with rawpy.imread(path) as raw:
                        Z[:, i] = raw.raw_image[y_coords, x_coords]
                else:
                    img = Image.open(path).convert("L")
                    Z[:, i] = np.array(img)[y_coords, x_coords]
            
            
            self.stocker_matrices(Z, exposures, file_path)
        
        exposures = np.array(exposures, dtype=np.float32)
        # Weighting function (Debevec)
        w = [z if z <= 0.5*(self.__bit_per_sample - 1) else (self.__bit_per_sample - 1)-z for z in range(self.__bit_per_sample)]
        # B is log delta t
        B = exposures
        l = 100
        
        solver = RC.ResponseCalculator()
        response_dic = {}
        inv_response_dic = {}
        
        if self.use_matlab:
            g, lE = solver.gsolve_matlab(Z, B, l, w)
            #g, lE = self.gsolve_python(Z, B, l, w)
        else:
            #responses = np.load('responses.npy')
            responses = np.power(np.linspace(0, 1, 1000), np.linspace(0.5, 1.5, 100)[:,None])
            for base_name in self.bases:
                print(f"Calcul de la réponse avec la base {base_name}...")
                E, complete_inv_response, complete_response = solver.get_response_params(Z, B, responses, base_name, n_params=10)
                response_dic[base_name] = complete_response
                inv_response_dic[base_name] = complete_inv_response
        
        exposures_values = np.linspace(0, 1, 1000)
        pixel_values = np.arange(65536)
        
        # Configuration de la figure avec une taille adaptée pour deux graphiques
        plt.figure(figsize=(16, 6))

        # --- PREMIER GRAPHIQUE ---
        plt.subplot(1, 2, 1)
        #plt.scatter((E[:, None] * B[None, :]).flatten(), Z.flatten(), marker='+')
        for base_name, complete_response in response_dic.items():
            plt.plot(exposures_values, complete_response, linewidth=2)
        plt.title(f"OECF - {'MATLAB' if self.use_matlab else 'Python'} ({self.__bit_per_sample} levels)")
        plt.xlabel("Log Exposure (E*dt)")
        plt.ylabel("Log Pixel Value (Z)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)

        # --- SECOND GRAPHIQUE ---
        plt.subplot(1, 2, 2)
        #plt.scatter(Z.flatten(), (E[:, None] * B[None, :]).flatten(), marker='+')
        for base_name, complete_inv_response in inv_response_dic.items():
            plt.plot(pixel_values, complete_inv_response, linewidth=2)
        plt.title(f"Inverse OECF - {'MATLAB' if self.use_matlab else 'Python'}")
        plt.xlabel("Log Pixel Value (Z)")
        plt.ylabel("Log Exposure (E*dt)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)

        # Ajuster automatiquement l'espacement entre les graphiques
        plt.tight_layout()
        plt.show()
        
        # Traitement de photos
        self.stock_functions("nikon_arctan_responses.npz", complete_inv_response, complete_response)
        # resinv, res = self.load_functions("nikon_arctan_responses.npz") 
        # with rawpy.imread("C:\\Users\\benja\\Documents\\3A\\Stray-light-elimination\\OECF\\SonyA6700(2)\\SHUTTERS00032.ARW") as raw:
        #     raw_data = raw.raw_image.copy()
        #     imageio.imsave('raw_ref.tiff', raw_data.astype(np.uint16))
        #     img_linear = self.apply_linearization(raw_data, resinv)
        #     imageio.imsave('raw_linear.tiff', raw_data.astype(np.uint16))
        #     img_final = self.apply_response(img_linear, res)
        #     imageio.imsave('raw_final.tiff', img_final.astype(np.uint16))
    
    def stocker_matrices(self, Z, exposures, filename):
        try:
            np.savetxt(filename + "_Z.txt", Z, fmt="%d", header=f"Matrice Z ({Z.shape[0]}x{Z.shape[1]})")
            np.savetxt(filename + "_exposures.txt", exposures, fmt="%.10f", header=f"Exposition en fréquence par image")
            print(f"Matrices stockées avec succès dans {filename}")
        except Exception as e:
            print(f"Erreur lors du stockage : {e}")

    def charger_matrices(self, filename):
        try:
            Z = np.loadtxt(filename + "_Z.txt")
            exposures = np.loadtxt(filename + "_exposures.txt")
            print(f"Matrices chargées avec succès.")
            return Z, exposures
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return None, None
       
    def getG(self):
        return self.__g

    def getTag(self):
        return self.__tag
 
    def getListExpo(self):
        return self.__exposures
    
    def apply_linearization(self,image_raw, inv_resp_lut):
        """
        Passe l'image brute dans l'espace linéaire [0, 1]
        inv_resp_lut: votre tableau 'complete_inv_response'
        """
        # On utilise l'image comme index dans la table de correspondance
        # C'est extrêmement rapide
        return inv_resp_lut[image_raw.astype(int)]
    
    def apply_response(self, image_linear, resp_vals, linear_vals=np.linspace(0, 1, 1000)):
        """
        Passe l'image linéaire [0, 1] vers l'espace non-linéaire
        linear_vals: np.linspace(0, 1, 1000)
        resp_vals: votre tableau 'complete_response'
        """
        # Ici on utilise np.interp car l'image linéaire contient des flottants
        flat_img = image_linear.flatten()
        interp_flat = np.interp(flat_img, linear_vals, resp_vals)
        return interp_flat.reshape(image_linear.shape)
    
    def stock_functions(self, filename, complete_inv_response, complete_response):
        """
        Sauvegarde les tableaux de réponse dans un fichier compressé .npz.
        """
        # On force l'extension si elle n'est pas présente
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        np.savez_compressed(filename, 
                            inv_res=complete_inv_response, 
                            res=complete_response)
        print(f"Fonctions sauvegardées avec succès dans {filename}")

    def load_functions(self, filename):
        """
        Charge les tableaux de réponse à partir du fichier .npz.
        Retourne (complete_inv_response, complete_response)
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        data = np.load(filename)
        
        complete_inv_response = data['inv_res']
        complete_response = data['res']
        
        return complete_inv_response, complete_response
        