import json
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import torch
import numpy as np
import faiss
import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image_pairs(data_file, image_folder):
    """
    Load paired images from the dataset JSON and image folder.
    """
    with open(data_file, 'r') as f:
        data = json.load(f)

    image_pairs = []
    captions = []

    for item in data['images']:
        img_id = item['imgid']
        file_A = os.path.join(image_folder, 'A', item['filename'])
        file_B = os.path.join(image_folder, 'B', item['filename'])

        if os.path.exists(file_A) and os.path.exists(file_B):
            image_pairs.append((img_id, file_A, file_B))

        for sentence in item['sentences']:
            captions.append({'img_id': img_id, 'caption': sentence['raw']})

    return image_pairs, captions

def encode_images(image_pairs, model, preprocess, device):
    """
    Encode image pairs into feature vectors.
    """
    image_features = []
    img_ids = []

    for img_id, img_A_path, img_B_path in tqdm(image_pairs, desc="Encoding images"):
        image_A = preprocess(Image.open(img_A_path)).unsqueeze(0).to(device)
        image_B = preprocess(Image.open(img_B_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            feature_A = model.encode_image(image_A)[0]  # 提取第一个值
            feature_B = model.encode_image(image_B)[0]  # 提取第一个值

        diff_feature = (feature_B - feature_A).cpu().numpy()
        image_features.append(diff_feature)
        img_ids.append(img_id)

    return np.vstack(image_features), img_ids


def encode_captions(captions, model, device):
    """
    Encode captions into feature vectors.
    """
    caption_features = []
    caption_texts = []
    caption_img_ids = []

    for caption in tqdm(captions, desc="Encoding captions"):
        caption_text = caption['caption']
        img_id = caption['img_id']

        with torch.no_grad():
            tokenized_text = clip.tokenize([caption_text]).to(device)
            feature = model.encode_text(tokenized_text).cpu().numpy()

        caption_features.append(feature)
        caption_texts.append(caption_text)
        caption_img_ids.append(img_id)

    return np.vstack(caption_features), caption_texts, caption_img_ids

def retrieve_similar_captions(image_features, caption_features, caption_texts, img_ids, caption_img_ids, k=5):
    """
    Retrieve k most similar captions for each image based on feature similarity.
    """
    # Ensure correct data type
    image_features = image_features.astype(np.float32)
    caption_features = caption_features.astype(np.float32)

    faiss.normalize_L2(image_features)
    faiss.normalize_L2(caption_features)

    index = faiss.IndexFlatIP(caption_features.shape[1])
    index.add(caption_features)

    _, neighbors = index.search(image_features, k)

    results = {}
    for img_idx, img_id in enumerate(img_ids):
        img_neighbors = []
        for neighbor_idx in neighbors[img_idx]:
            if caption_img_ids[neighbor_idx] != img_id:  # Exclude self captions
                img_neighbors.append(caption_texts[neighbor_idx])
            if len(img_neighbors) == k:
                break

        results[img_id] = img_neighbors

    return results


def main():
    data_file = 'data/LEVIR_CC/LevirCCcaptions.json'  # Path to dataset JSON
    image_folder = 'data/LEVIR_CC/images/train'  # Path to image folder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("Loading data...")
    image_pairs, captions = load_image_pairs(data_file, image_folder)

    if not image_pairs:
        raise ValueError("No image pairs found. Check the data and file paths.")

    print("Encoding images...")
    image_features, img_ids = encode_images(image_pairs, model, preprocess, device)

    print("Encoding captions...")
    caption_features, caption_texts, caption_img_ids = encode_captions(captions, model, device)

    print("Retrieving similar captions...")
    results = retrieve_similar_captions(image_features, caption_features, caption_texts, img_ids, caption_img_ids)

    output_file = './retrieved_captions.json'
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
