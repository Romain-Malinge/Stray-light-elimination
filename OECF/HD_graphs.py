import os
import rawpy
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import HDData as hd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

SUPPORTED_EXTENSIONS = (".arw", ".jpg", ".jpeg", ".png")


class PhotoViewer:
    
    def __init__(self, root, folder_path):
        self.__id_im = 0;
        self.root = root
        self.root.title("Liseuse Photo")

        self.folder_path = folder_path
        self.files = self.load_images(folder_path)
        self.index = 0

        self.canvas = tk.Canvas(self.root, width=1000, height=800, bg="black")
        self.canvas.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        tk.Button(
            button_frame, 
            text="< Précédent", 
            command=self.prev_image,
            bg="#f0f0f0",          
            fg="#333333",          
            activebackground="#e0e0e0",
            font=("Arial", 10, "bold"),
            relief="flat",         
            borderwidth=0,
            cursor="hand2",        
            padx=15,
            pady=8
        ).pack(side="left", padx=10, pady=10)
        tk.Button(
            button_frame, 
            text="Suivant >", 
            command=self.next_image,
            bg="#f0f0f0",         
            fg="#333333",         
            activebackground="#e0e0e0",
            font=("Arial", 10, "bold"),
            relief="flat",        
            borderwidth=0,
            cursor="hand2",        
            padx=15,
            pady=8
        ).pack(side="right", padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.calcul_oecf)

        self.show_image()

    def load_images(self, folder):
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

    def load_image(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".arw":
            with rawpy.imread(path) as raw:
                
                # Dimensions de l'image RAW et RGB (post-processée)
                self.raw_height, self.raw_width = raw.raw_image_visible.shape
                self.rgb_width, self.rgb_height = raw.sizes.width, raw.sizes.height
                
                self.__bits = raw.white_level + 1
                
                # Les marges du capteur (zones non actives)
                self.offset_top = raw.sizes.top_margin
                self.offset_left = raw.sizes.left_margin
                
                # Conserver une vue de l'image RAW
                self.generer_vue_raw(raw)
                
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                    no_auto_bright=False,
                    half_size=True)
                
            img = Image.fromarray(rgb)
            
            # Redimensionnement pour l'affichage 
            screen_max_size = (1000, 1000)
            img_display = img.copy()
            img_display.thumbnail(screen_max_size, Image.Resampling.LANCZOS)
            self.display_width, self.display_height = img_display.size
            
        else:
            print("Extension non supportée pour l'affichage :", ext)
            return None
        
        return img_display
    
    def generer_vue_raw(self, raw):
        # Récupérer la matrice brute
        data = raw.raw_image.astype(np.float32)
        
        # Normalisation rapide pour la visibilité (0-255)
        # On soustrait le noir et on scale vers 8 bits
        black = np.min(raw.black_level_per_channel)
        white = raw.white_level
        data = (data - black) / (white - black) * 255
        data = np.clip(data, 0, 255).astype(np.uint8)
        
        # Créer une image PIL (L = Luminance / Grayscale)
        self.__image_raw_full = Image.fromarray(data, mode='L')

    def show_image(self):
        if not self.files:
            return

        self.canvas.delete(self.__id_im)

        path = self.files[self.index]
        img = self.load_image(path)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.__id_im = self.canvas.create_image(
            0, 
            0, 
            anchor="nw", 
            image=self.tk_image,
            tags="background")
        
        self.canvas.tag_lower("background")

        self.root.title(f"{os.path.basename(path)}")

    def calcul_oecf(self, event):

        print("Veuillez patienter pendant le calcul de la courbe...")
        hd.HDData(self.folder_path, self.raw_height, self.raw_width, self.__bits)

    def next_image(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

## Boucle principale
def main():
    root = tk.Tk()
    folder = filedialog.askdirectory(title="Choisir un dossier d'images")
    if not folder:
        return

    app = PhotoViewer(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
