import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import rawpy


# Fonction d'affichage des masques, points et boîtes

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# Fonction de création du prédicteur SAM
def create_sam_predictor():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = os.path.abspath("./model/sam2.1_hiera_small.pt")
    model_cfg = os.path.abspath("./model/sam2.1_hiera_s.yaml")

    sam2_model = build_sam2(
        config_file=model_cfg,
        ckpt_path=sam2_checkpoint,
        device=device
    )

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, device


# Fonction de conversion d'une image en RGB et redimensionnement si nécessaire
def rgb_resize(image, taille_max):

    # Cas image PIL
    if isinstance(image, Image.Image):
        # Conversion explicite en RGB quel que soit le mode
        if image.mode != "RGB":
            image = image.convert("RGB")
    else:
        raise TypeError("L'entrée doit être une image PIL")

    # Redimensionnement
    if max(image.size) > taille_max:
        scale = taille_max / max(image.size)
        new_size = (
            int(image.size[0] * scale),
            int(image.size[1] * scale)
        )
        image = image.resize(new_size, Image.BICUBIC)

    # Conversion numpy uint8
    image = np.asarray(image, dtype=np.uint8).copy()

    return image



# Fonction principale pour créer des masques pour toutes les images dans un dossier
def make_mask(folder_path, taille_max=2048):
    # Charger toute les images du dossier
    images = []
    images_names = os.listdir(folder_path)
    idx = 0
    nb_im = len(images_names)
    for image_name in images_names:
        idx += 1
        image_path = os.path.join(folder_path, image_name)

        # Cas RAW
        if isinstance(image_name, str) and image_name.lower().endswith(".nef"):
            with rawpy.imread(image_path) as raw:
                image = raw.postprocess(
                    use_camera_wb=True,
                    bright=1.0,
                    output_bps=8
                )
                image = Image.fromarray(image)
        # Cas autres formats
        else:
            image = Image.open(image_path)

        image = rgb_resize(image, taille_max)
        images.append(image)
        print(f"[{idx}/{nb_im}] Resized image {image_name} to {image.shape[1]} x {image.shape[0]}")
    
    # Créer un nouveau dossier pour sauvegarder les images segmentées à coté du premier dossier
    output_folder = os.path.join(os.path.dirname(folder_path), os.path.basename(folder_path) + '_segmented')
    os.makedirs(output_folder, exist_ok=True)

    predictor, device = create_sam_predictor()

    im1 = images[0]
    predictor.set_image(im1)

    H, W, _ = im1.shape
    input_point = np.array([[W//2, H//2]])
    input_label = np.array([1])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Convertir le masque en format uint8
    mask = masks[0].astype(np.uint8)

    # Garder la composante connexe la plus grande du masque
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        mask = (labels_im == largest_label).astype(np.uint8)
    
    # Sauvegarder chaque image avec le masque appliqué en format PNG
    idx = 0
    for image, image_name in zip(images, images_names):
        idx += 1
        # Appliquer le masque à l'image
        image = image * mask[:, :, None]
        alpha = (mask * 255).astype(np.uint8)[:, :, None]
        image_rgba = np.concatenate((image, alpha), axis=2)

        # retirer l'extension du nom de l'image
        image_name = os.path.splitext(image_name)[0]
        save_path = os.path.join(output_folder, image_name + '_segmented.png')

        Image.fromarray(image_rgba, mode='RGBA').save(save_path)
        print(f"[{idx}/{nb_im}] image saved to {save_path}")


if __name__ == "__main__":

    # INPUTS
    folder = './dataset/validation/bouddha/'
    taille_max = 2048

    # Liste des sous-dossiers dans le dossier
    folders = os.listdir(folder)
    # Garder les folders seulement commencent par S et ne fissant pas par _segmented
    folders = [f for f in folders if f.startswith('S') and not f.endswith('_segmented')]
    for f in folders:
        folder_path = os.path.join(folder, f)
        print(f"\nProcessing folder: {folder_path}")

        # Create masks for all images in the folder
        make_mask(folder_path, taille_max)
    
    print("\nDone.\n")