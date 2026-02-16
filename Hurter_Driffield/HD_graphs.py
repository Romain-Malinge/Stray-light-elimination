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

SUPPORTED_EXTENSIONS = (".arw", ".jpg", ".jpeg", ".png")


class PhotoViewer:
    
    def __init__(self, root, folder_path):
        self.__ids_markers = 0
        self.__rad_marker_hit_box = 15
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

        tk.Button(button_frame, text="⬅ Précédent", command=self.prev_image).pack(side="left", padx=10)
        tk.Button(button_frame, text="Suivant ➡", command=self.next_image).pack(side="left", padx=10)

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
                self.raw_height, self.raw_width = raw.raw_image.shape
                self.rgb_width, self.rgb_height = raw.sizes.width, raw.sizes.height
                
                # Les marges du capteur (zones non actives)
                self.offset_top = raw.sizes.top_margin
                self.offset_left = raw.sizes.left_margin
                
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                    no_auto_bright=False)
                
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

        # Conversion coordonnées affichage → coordonnées originales
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
            print("Veuillez patienter pendant le calcul de la courbe H&D...")
            
            # Ajouter la courbe correspondante dans le graphique
            hddata = hd.HDData(self.folder_path, x_real, y_real, tag_maker_text)
            self.__graph.add_graph(hddata)

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
        self.ax.set_xlabel("Log10(Exposure Time [s])")
        self.ax.set_ylabel("Valeur Numérique Linéaire (DN)")
        self.ax.set_xlim(-4, 1)     # 1/10000 → 10 sec approx
        self.ax.set_ylim(0, 17000)

        self.ax.grid(True)

        # Intégration dans Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_graph(self, hddata: hd.HDData):
        line, = self.ax.plot(hddata.getListExpo(), hddata.getListPixValues(), 'o-', label=hddata.getTag())
        self.__list_pairs[hddata.getTag()] = line
        self.ax.legend(
            loc="best",
            fontsize=10,
            frameon=True)
        self.canvas.draw()
    
    def remove_graph(self, tag):
        if tag in self.__list_pairs:
            # Enlever du graphique
            self.__list_pairs[tag].remove()
            # Enlever de la liste
            del self.__list_pairs[tag]
            self.ax.legend(
                loc="best",
                fontsize=10,
                frameon=True)
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
