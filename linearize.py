import numpy as np
import rawpy
from PIL import Image
import os
import matplotlib.pyplot as plt


def apply_linearization(image_raw, inv_resp_lut):
    """
    Passe l'image brute dans l'espace linéaire [0, 1]
    inv_resp_lut: votre tableau 'complete_inv_response'
    """
    return inv_resp_lut[image_raw.raw_image_visible.astype(int)]


def apply_response(image_raw_linear, resp_vals, linear_vals=None):

    """
    Passe l'image linéaire [0, 1] vers l'espace non-linéaire
    linear_vals: np.linspace(0, 1, 1000)
    resp_vals: votre tableau 'complete_response'
    """
    # Ici on utilise np.interp car l'image linéaire contient des flottants
    if linear_vals is None:
        linear_vals = np.linspace(0, 1, len(resp_vals))
    flat_img = image_raw_linear.flatten()
    interp_flat = np.interp(flat_img, linear_vals, resp_vals)
    return interp_flat.reshape(image_raw_linear.shape)


def stock_functions(filename, complete_inv_response, complete_response):
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


def load_functions(filename):
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