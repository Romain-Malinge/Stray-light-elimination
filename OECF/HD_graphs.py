import os
import rawpy
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import Markers as mk
import HDData as hd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

SUPPORTED_EXTENSIONS = (".arw", ".jpg", ".jpeg", ".png")


class PhotoViewer:
    
    def __init__(self, root, folder_path):
        self.__ids_markers = 0
        self.__rad_marker_hit_box = 20
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

        self.canvas.bind("<Button-1>", self.manip_marker)

        self.markers = []
        self.show_image()
        
        self.__graph = HDGraphWindow(self.root)

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

    def manip_marker(self, event):
        
        # Coordonnées du clic dans le canevas
        x_display = event.x
        y_display = event.y

        # Conversion coordonnées affichage en coordonnées originales
        scale_x = self.rgb_width / self.display_width
        scale_y = self.rgb_height / self.display_height
        
        x_real = int(x_display * scale_x + self.offset_left)
        y_real = int(y_display * scale_y + self.offset_top)
        
        print(f"Clic écran: ({x_display}, {y_display}) -> Coordonnée RAW: ({x_real}, {y_real})")
        
        # Vérifier si un marqueur existe déjà à proximité
        already_exist = False
        marker = None
        for marker in self.markers:
            posX = marker.getCordX()
            posY = marker.getCordY()
            
            if (abs(posX - x_real) <= self.__rad_marker_hit_box) and (abs(posY - y_real) <= self.__rad_marker_hit_box):
                already_exist = True
                break
           
        # Si un marqueur existe déjà à proximité, le supprimer 
        if already_exist :
            self.markers.remove(marker)
            self.canvas.delete(marker.getTag())
            print(f"Suppresion du marqeur d'id : {marker.getTag()}")
            
            # Supprimer la courbe correspondante dans le graphique
            self.__graph.remove_graph(marker.getTag())
            
        # Sinon, en créer un nouveau
        else:
            self.__ids_markers += 1
            tag_maker_text = "M_" + str(self.__ids_markers)
            
            # Dessiner un petit cercle rouge
            r = 3
            self.canvas.create_oval(
                x_display - r, y_display - r,
                x_display + r, y_display + r,
                fill="red",
                tags=tag_maker_text
            )
            
            self.canvas.create_text(
                x_display,
                y_display - 12,
                text=tag_maker_text,
                fill="red",
                font=("Arial", 10, "bold"),
                tags=tag_maker_text
            )

            newMark = mk.Marker(x_real, y_real, tag_maker_text)
            self.markers.append(newMark)
            
            print(f"Ajout du marqueur d'ID {tag_maker_text}")
            print("Veuillez patienter pendant le calcul de la courbe...")
            
            # Ajouter la courbe correspondante dans le graphique
            hddata = hd.HDData(self.folder_path, x_real, y_real, tag_maker_text, self.raw_height, self.raw_width)
            #self.__graph.add_graph(hddata)
            print(" : Courbe ajoutée pour le marqueur ", tag_maker_text)

    def next_image(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()
    
class HDGraphWindow:
    
    def __init__(self, parent):

        # Nouvelle fenêtre
        self.window = tk.Toplevel(parent)
        self.window.title("Courbe Opto-Electronic Conversion Function (OECF)")
        self.window.geometry("800x600")
        
        # Dictionnaire pour stocker les courbes associées à chaque marqueur
        self.__list_pairs = {}

        # Création figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(7, 5))

        # Configuration des axes
        self.ax.set_title("Courbe OECF")
        self.ax.set_xlabel("Temps d'exposition (s)")
        self.ax.set_ylabel("Valeur Numérique RAW")
        
        self.ax.set_ylim(1, 20000)
        self.ax.set_xscale('log', base=10)
        self.ax.set_yscale('log', base=10)
        
        self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.ax.yaxis.get_major_formatter().set_scientific(False)
        
        def format_func(value, tick_number):
            if value >= 1:
                return f"{value:.1f}s"
            else:
                return f"1/{int(round(1/value))}"
        
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        stops = [1/8000, 1/2000, 1/500, 1/125, 1/30, 1/8, 0.5, 2, 4]
        self.ax.set_xticks(stops)

        self.ax.grid(True, which="both", linestyle='--', alpha=0.5)

        # Intégration dans Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_graph(self, hddata: hd.HDData):
        # On ajoute la courbe au graphique
        line, = self.ax.plot(hddata.getListExpo(), hddata.getG(), 'o-', label=hddata.getTag())
        # On stocke la courbe dans le dictionnaire pour pouvoir la supprimer plus tard
        self.__list_pairs[hddata.getTag()] = line
        # Mise à jour de la légende
        self.ax.legend(
            loc="best",
            fontsize=10,
            frameon=True)
        # Mise à jour de l'affichage
        self.canvas.draw()
    
    def remove_graph(self, tag):
        if tag in self.__list_pairs:
            # Enlever la courbe du graphique
            self.__list_pairs[tag].remove()
            # Enlever la courbe de la liste
            del self.__list_pairs[tag]
            # Mettre à jour la légende
            if self.__list_pairs:
                self.ax.legend(
                loc="best",
                fontsize=10,
                frameon=True)
            # Mise à jour de l'affichage
            self.canvas.draw()


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
