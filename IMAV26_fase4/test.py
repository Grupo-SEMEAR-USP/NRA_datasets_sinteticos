import os
import cv2
import glob
import random
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES
# ============================================================
IMAGES_DIR = r"/home/joao/dataset_fase4_imav_26/images"
LABELS_DIR = r"/home/joao/dataset_fase4_imav_26/labels"

# Mapeamento de IDs para nomes de classes
CLASS_NAMES = {
    0: "bombeiro"
}

def draw_yolo_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # OpenCV lê em BGR; converte para RGB para exibir corretamente no Matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, bw, bh = map(float, parts)
            class_id = int(class_id)

            # Converter coordenadas relativas (0.0 - 1.0) para pixels na imagem
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            # Clamping para manter os limites dentro da imagem
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Cor da caixa (Verde) e nome da classe
            color = (0, 255, 0)
            label_name = CLASS_NAMES.get(class_id, f"ID: {class_id}")

            # Desenha o retângulo e o rótulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, 
                label_name, 
                (x1, max(y1 - 8, 15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

    return img

def visualize_dataset(num_samples=6, shuffle=True):
    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    
    if not image_paths:
        print(f"Nenhuma imagem encontrada em {IMAGES_DIR}")
        return

    if shuffle:
        random.shuffle(image_paths)

    sample_paths = image_paths[:num_samples]

    # Cria uma grade de 2 linhas x 3 colunas
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, img_path in enumerate(sample_paths):
        filename = os.path.basename(img_path)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        annotated_img = draw_yolo_boxes(img_path, label_path)

        if annotated_img is not None:
            axes[idx].imshow(annotated_img)
            axes[idx].set_title(filename, fontsize=10)
            axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Caso não tenha matplotlib ou opencv instalados, rode no terminal:
    # pip install opencv-python matplotlib
    visualize_dataset(num_samples=6, shuffle=True)