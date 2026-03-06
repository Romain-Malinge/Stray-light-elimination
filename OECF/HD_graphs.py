import os
import rawpy
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import HDData as hd
import ResponseCalculator as rc

class PhotoViewer:
    
    def __init__(self, root, folder_path):
        self.__id_im = 0
        self.root = root
        self.root.title("Liseuse Photo")

        self.folder_path = folder_path
        self.files = self.load_images(folder_path)
        self.index = 0

        self.canvas = tk.Canvas(self.root, width=1000, height=800, bg="black")
        self.canvas.pack()

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        self.use_matlab_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Utiliser MATLAB Engine", variable=self.use_matlab_var).pack()
        
        # 1. Création du Menubutton (le bouton qui déclenche le menu)
        mb = tk.Menubutton(control_frame, text="Choix des bases", relief=tk.RAISED)
        mb.pack(pady=10)

        # 2. Création du Menu attaché au Menubutton
        menu_deroulant = tk.Menu(mb, tearoff=0)
        mb["menu"] = menu_deroulant
        self.choix_bases = {}
        
        for base in rc.BASES:
            var = tk.BooleanVar()
            self.choix_bases[base] = var
            menu_deroulant.add_checkbutton(
                label=base, 
                variable=var, 
                onvalue=True, 
                offvalue=False
            )

        button_frame = tk.Frame(control_frame)
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
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) 
                if f.lower().endswith(hd.SUPPORTED_EXTENSIONS)]

    def load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".arw" or ext == ".nef":
            with rawpy.imread(path) as raw:
                self.raw_height, self.raw_width = raw.raw_image_visible.shape
                self.__bits = raw.white_level + 1
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
            img = Image.fromarray(rgb)
        else:
            img_full = Image.open(path)
            self.raw_width, self.raw_height = img_full.size
            self.__bits = 256
            img = img_full.convert("RGB")
            
        img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        self.display_width, self.display_height = img.size
        return img

    def show_image(self):
        if not self.files: return
        self.canvas.delete("all")
        img = self.load_image(self.files[self.index])
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.root.title(f"{os.path.basename(self.files[self.index])}")

    def calcul_oecf(self, event):
        selectionnes = [nom for nom, var in self.choix_bases.items() if var.get()]
        hd.HDData(self.folder_path, self.raw_height, self.raw_width, self.__bits, use_matlab=self.use_matlab_var.get(), bases = selectionnes)

    def next_image(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

def main():
    root = tk.Tk()
    folder = filedialog.askdirectory()
    if folder:
        PhotoViewer(root, folder)
        root.mainloop()

if __name__ == "__main__":
    main()