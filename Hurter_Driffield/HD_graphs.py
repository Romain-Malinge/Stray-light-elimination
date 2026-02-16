import os
import rawpy
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import Markers as mk
import exifread
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = (".arw", ".jpg", ".jpeg", ".png")


class PhotoViewer:
    
    def __init__(self, root, folder_path):
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
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                    no_auto_bright=False)
                
            img = Image.fromarray(rgb)
        else:
            img = Image.open(path)

        self.original_width, self.original_height = img.size

        img.thumbnail((1000, 800))
        self.display_width, self.display_height = img.size

        return img

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
        x_display = event.x
        y_display = event.y

        # Conversion coordonnées affichage → coordonnées originales
        scale_x = self.original_width / self.display_width
        scale_y = self.original_height / self.display_height
        
        x_real = int(x_display * scale_x)
        y_real = int(y_display * scale_y)
        
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
            self.canvas.delete(marker.getID())
            print(f"Suppresion du marqeur d'id : {marker.getID()}")
            
            # Supprimer la courbe correspondante dans le graphique
            self.__graph.remove_graph(marker.getID())
            
        # Sinon, en créer un nouveau
        else:

            # Dessiner un petit cercle rouge
            r = 3
            id = self.canvas.create_oval(
                x_display - r, y_display - r,
                x_display + r, y_display + r,
                fill="red"
            )
            
            newMark = mk.Marker(x_real, y_real, id)
            self.markers.append(newMark)
            
            print(f"Ajout du marqueur d'ID {id}")
            
            # Ajouter la courbe correspondante dans le graphique
            hddata = HDData(self.folder_path, x_real, y_real, id)
            self.__graph.add_graph(hddata)

    def next_image(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

class HDData:
    
    def __init__(self, folder_path, x, y, id):
        self.__folder_path = folder_path
        self.__x = x
        self.__y = y
        self.__id = id
        
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
                raw_data = raw.raw_image_visible

                # Sécurité
                if self.__y >= raw_data.shape[0] or self.__x >= raw_data.shape[1]:
                    continue

                value = raw_data[self.__y, self.__x] - raw.black_level_per_channel[0]

            exposures.append(exposure_time)
            pixel_values.append(value)

        # Convertir en log10
        exposures_log = [math.log10(t) for t in exposures]

        # Trier par exposition croissante
        combined = sorted(zip(exposures_log, pixel_values), key=lambda x: x[0])

        exposures_log_sorted = [c[0] for c in combined]
        pixel_values_sorted = [c[1] for c in combined]

        return exposures_log_sorted, pixel_values_sorted
       
    def getListExpo(self):
        return self.__exposures
    
    def getListPixValues(self):
        return self.__pixel_values

    def getID(self):
        return self.__id
    
class HDGraphWindow:
    def __init__(self, parent):

        # Nouvelle fenêtre
        self.window = tk.Toplevel(parent)
        self.window.title("Courbe Hurter & Driffield")
        self.window.geometry("800x600")
        
        self.__list_pairs = {}

        # Création figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(7, 5))

        # Configuration des axes
        self.ax.set_title("Courbe de Hurter & Driffield")
        self.ax.set_xlabel("log10(Temps de pose [s])")
        self.ax.set_ylabel("Signal RAW normalisé")
        self.ax.set_xlim(-4, 1)     # 1/10000 → 10 sec approx
        self.ax.set_ylim(0, 17000)

        self.ax.grid(True)

        # Intégration dans Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_graph(self, hddata: HDData):
        line, = self.ax.plot(hddata.getListExpo(), hddata.getListPixValues(), 'o-')
        self.__list_pairs[hddata.getID()] = line
        
        self.canvas.draw()
    
    def remove_graph(self, id):
        if id in self.__list_pairs:
            self.__list_pairs[id].remove()
            del self.__list_pairs[id]
            self.canvas.draw()

def main():
    root = tk.Tk()
    folder = filedialog.askdirectory(title="Choisir un dossier d'images")
    if not folder:
        return

    app = PhotoViewer(root, folder)

    root.mainloop()


if __name__ == "__main__":
    main()
