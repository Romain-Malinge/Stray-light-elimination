import exifread
import os
import rawpy
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr
from PIL import Image
import math

# Import MATLAB if possible
try:
    import matlab.engine
except ImportError:
    print("Warning: matlab.engine not found. MATLAB mode will fail.")

def afficher_progression(actuel, total, nom_fichier=""):
    largeur_barre = 40
    progression = actuel / total
    nb_carres = int(largeur_barre * progression)
    barre = "█" * nb_carres + "░" * (largeur_barre - nb_carres)
    pourcentage = int(progression * 100)
    sys.stdout.write(f"\r|{barre}| {pourcentage}% - Analyse : {nom_fichier}")
    sys.stdout.flush()

class HDData:
    
    def __init__(self, folder_path, height, width, bits, use_matlab=False):
        self.__folder_path = folder_path
        self.__height = height
        self.__width = width
        self.__bit_per_sample = bits
        self.use_matlab = use_matlab
        
        # Load manual exposure table if it exists
        self.__manual_exposures = self._load_exposure_table()
        
        self.__g, self.__lE, self.__exposures = self.extract_hd_data()
    
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
                exposure = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
                
                if exposure:
                    if "/" in str(exposure):
                        num, den = str(exposure).split("/")
                        return int(num) / int(den)
                    return float(str(exposure))
        except:
            pass
            
        print(f"\nAttention: Pas d'exposition trouvée pour {filename}. Valeur par défaut 1.0 utilisée.")
        return 1.0 
            
    def gsolve_matlab(self, Z, B, l, w):
        print("\nStarting MATLAB Engine...")
        eng = matlab.engine.start_matlab()
        
        Z_mat = matlab.double(Z.tolist())
        B_mat = matlab.double(np.array(B).reshape(-1,1).tolist())
        l_mat = float(l)
        w_mat = matlab.double(np.array(w).tolist())
        n_mat = float(self.__bit_per_sample)
        
        print("Running gsolve.m in MATLAB...")
        g, lE = eng.gsolve(Z_mat, B_mat, l_mat, w_mat, n_mat, nargout=2)
        
        eng.quit()
        return np.array(g).flatten(), np.array(lE).flatten()

    def gsolve_python(self, Z, B, l, w):
        n = self.__bit_per_sample 
        Z1 = np.size(Z, 0) 
        Z2 = np.size(Z, 1) 
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
        
        A[k, n//2] = 1
        k += 1

        for i in range(n-2):
            A[k, i]   =    l*w[i+1]
            A[k, i+1] = -2*l*w[i+1]
            A[k, i+2] =    l*w[i+1]
            k += 1
        
        A = A.tocsr()
        x, istop, itn = lsqr(A, b)[:3]
        return x[:n], x[n:]
            
    def extract_hd_data(self):
        valid_exts = (".arw", ".png", ".jpg", ".jpeg")
        files = sorted([
            os.path.join(self.__folder_path, f)
            for f in os.listdir(self.__folder_path)
            if f.lower().endswith(valid_exts) and f.lower() != "exposure.txt"
        ])
        
        if not files:
            return None, None, None

        nb_files = len(files)
        num_samples = 50
        Z_file_path = os.path.join(self.__folder_path, f"Z_{num_samples}_{nb_files}_.txt")
        
        if os.path.isfile(Z_file_path):
            Z, exposures = self.charger_matrice_Z(Z_file_path)
        else:
            y_coords = np.random.randint(0, self.__height, num_samples)
            x_coords = np.random.randint(0, self.__width, num_samples)
            exposures = []
            Z = np.zeros((num_samples, nb_files))
            
            for i, path in enumerate(files):
                afficher_progression(i+1, nb_files, nom_fichier=os.path.basename(path))
                exposure_time = self.extract_parameters(path)
                exposures.append(exposure_time) 
                
                if path.lower().endswith(".arw"):
                    with rawpy.imread(path) as raw:
                        Z[:, i] = raw.raw_image[y_coords, x_coords]
                else:
                    img = Image.open(path).convert("L")
                    Z[:, i] = np.array(img)[y_coords, x_coords]
            
            print("")
            self.stocker_matrice_Z(Z, exposures, Z_file_path)
        
        exposures = np.array(exposures, dtype=np.float32)
        # Weighting function (Debevec)
        w = [z if z <= 0.5*(self.__bit_per_sample - 1) else (self.__bit_per_sample - 1)-z for z in range(self.__bit_per_sample)]
        # B is log delta t
        B = [np.log(e) for e in exposures]
        l = 100
        
        if self.use_matlab:
            g, lE = self.gsolve_matlab(Z, B, l, w)
        else:
            g, lE = self.gsolve_python(Z, B, l, w)
        
        pixel_values = np.arange(len(g)) 
        g_plot = [np.log10(np.exp(i)) for i in g]
        plt.figure(figsize=(10, 6))
        plt.plot(g_plot, pixel_values, color='blue' if self.use_matlab else 'green', linewidth=2)
        plt.title(f"OECF - {'MATLAB' if self.use_matlab else 'Python'} ({self.__bit_per_sample} levels)")
        plt.xlabel("Log Exposure (ln E*dt)")
        plt.ylabel("Pixel Value")
        plt.grid(True)
        plt.show()
        
        return g, lE, exposures
    
    def stocker_matrice_Z(self, Z, exposures, filename):
        np.savetxt(filename, Z, fmt="%d")
        np.savetxt(filename[:-4] + "_exposures.txt", exposures, fmt="%.10f")

    def charger_matrice_Z(self, filename):
        Z = np.loadtxt(filename)
        exposures = np.loadtxt(filename[:-4] + "_exposures.txt")
        return Z, exposures