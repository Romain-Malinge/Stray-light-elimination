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
def create_sam_predictor(version):
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

    list_checkpoint = [f for f in os.listdir(f"./IA/SAM_2/{version}") if f.endswith(".pt")]
    sam2_checkpoint = os.path.abspath(f"./IA/SAM_2/{version}/{list_checkpoint[0]}")

    list_model_cfg = [f for f in os.listdir(f"./IA/SAM_2/{version}") if f.endswith(".yaml")]
    model_cfg = os.path.abspath(f"./IA/SAM_2/{version}/{list_model_cfg[0]}")
    sam2_model = build_sam2(
        config_file=model_cfg,
        ckpt_path=sam2_checkpoint,
        device=device
    )

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, device


def load_image(image_path, brightness=1.0):
    # Cas RAW
    if isinstance(image_path, str) and image_path.lower().endswith(".nef"):
        with rawpy.imread(image_path) as raw:
            image = raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                gamma=(1, 1),
                bright=brightness,
                output_bps=8
            )
            image = np.array(image)
    # Cas autres formats
    else:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
    return image


# Fonction pour obtenir les points d'entrée de l'utilisateur
def get_input_points(img, max_size=1000):
    input_points = []
    input_labels = []

    # Image affichée (redimensionnée si nécessaire)
    display_img = img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img

        if event == cv2.EVENT_LBUTTONDOWN:

            input_points.append([x, y])
            input_labels.append(1)

            cv2.circle(display_img, (x, y), 4, (0, 255, 0), -1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            input_points.append([x, y])
            input_labels.append(0)

            cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)

    cv2.namedWindow("Select points", cv2.WINDOW_NORMAL)
    cv2.imshow("Select points", display_img)
    # Redimensionner la fenêtre si l'image est trop grande
    H, W = img.shape[:2]
    if H > max_size or W > max_size:
        scale = min(max_size / H, max_size / W)
        new_W = int(W * scale)
        new_H = int(H * scale)
        cv2.resizeWindow("Select points", new_W, new_H)
    else:
        cv2.resizeWindow("Select points", W, H)
    cv2.setMouseCallback("Select points", mouse_callback)

    while True:
        cv2.imshow("Select points", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Entrée = validation
        if key == 13:
            break

        # r = réinitialiser les points
        if key == ord('r'):
            input_points = []
            input_labels = []

    cv2.destroyAllWindows()

    if len(input_points) > 0:
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
    else:
        input_points = None
        input_labels = None

    return input_points, input_labels


# Fonction principale pour créer des masques pour toutes les images dans un dossier
def make_mask(folder_path, taille, version = "small"):

    # Créer un nouveau dossier pour sauvegarder les images segmentées à coté du premier dossier
    output_folder = os.path.join(os.path.dirname(folder_path), os.path.basename(folder_path) + '_segmented')
    os.makedirs(output_folder, exist_ok=True)

    # trouvrer les nom des images dans le dossier
    images_names = os.listdir(folder_path)

    # Calculler le masque de la première image du dossier
    image1_path = os.path.join(folder_path, images_names[0])
    image1 = load_image(image1_path, brightness=1.0)

    predictor, device = create_sam_predictor(version)
    predictor.set_image(image1)
    input_point, input_label = get_input_points(image1)

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Convertir le masque en format uint8
    mask = masks[0].astype(np.uint8)

    # Calculer le reshape a effectuer
    H = image1.shape[0]
    W = image1.shape[1]
    ys, xs = np.where(mask > 0)
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()

    ctr_x = (min_x + max_x) // 2
    ctr_y = (min_y + max_y) // 2
    h = max_y - min_y
    w = max_x - min_x
    if h > w:
        min_x = max(0, ctr_x - h // 2)
        max_x = min(W, min_x + h)
    elif w > h:
        min_y = max(0, ctr_y - w // 2)
        max_y = min(H, min_y + w)
    
    # Ajouter une marge autour du masque
    margin = 10  # Ajuster la valeur de la marge selon vos besoins
    min_x = max(0, min_x - margin)
    max_x = min(W, max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(H, max_y + margin)
        
    # reshape in a taille x taille image
    mask = mask[min_y:max_y, min_x:max_x]
    mask = cv2.resize(mask, (taille, taille), interpolation=cv2.INTER_NEAREST)

    # Garder la composante connexe la plus grande du masque
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        mask = (labels_im == largest_label).astype(np.uint8)
    
    # Comblet les trou du masque
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Sauvegarder un masque binaire pour chaque image
    mask_save_path = os.path.join(output_folder, 'mask.png')
    Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(mask_save_path)
    print(f"Mask saved to {mask_save_path}")
    
    idx = 0
    nb_im = len(images_names)
    # Sauvegarder les images redimensionnées dans une liste
    for image_name in images_names:
        idx += 1
        image_path = os.path.join(folder_path, image_name)

        image = load_image(image_path, brightness=1.0)
        image = image[min_y:max_y, min_x:max_x]
        image = cv2.resize(image, (taille, taille), interpolation=cv2.INTER_AREA)
        image_rgb = np.multiply(image, mask[:, :, np.newaxis])

        save_path = os.path.join(output_folder, image_name + '_segmented.png')
        Image.fromarray(image_rgb, mode='RGB').save(save_path)
        print(f"[{idx}/{nb_im}] Image {image_name} reframed.")
    
    return mask

def make_just_one_mask(image_path, version = "small"):

    predictor, device = create_sam_predictor(version)

    image = load_image(image_path, brightness=1.0)
    predictor.set_image(image)
    input_point, input_label = get_input_points(image)

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
    
    # export mask as png
    mask_save_path = os.path.join(os.path.dirname(image_path), 'mask.png')
    Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(mask_save_path)
    print(f"Mask saved to {mask_save_path}")

    return mask
    

if __name__ == "__main__":

    # INPUTS
    folder = './data/eve'
    taille = 1024
    sam_version = "large"

    make_mask(folder, taille, sam_version)

    
    print("Done.\n")