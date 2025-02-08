import json
import os
import torch
import numpy as np
import faiss
import clip
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

def filter_changed_captions(dataset_path):
    """
    读取 JSON 数据并筛选出 changeflag=1 的 captions
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    changed_captions = []
    caption_ids = []

    for item in dataset["images"]:
        if item["changeflag"] == 1:
            for sentence in item["sentences"]:
                changed_captions.append(" ".join(sentence["tokens"]))
                caption_ids.append(item["imgid"])

    return changed_captions, caption_ids


def filter_captions(captions, ids):
    """
    过滤 captions, 使其长度不超过 50
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    bs = 256

    encodings = []
    for idx in range(0, len(captions), bs):
        encodings += tokenizer.batch_encode_plus(
            captions[idx: idx + bs], return_tensors="np", padding=True
        )["input_ids"].tolist()

    filtered_captions = []
    filtered_image_ids = []

    for image_id, caption, encoding in zip(ids, captions, encodings):
        if len(encoding) <= 50:
            filtered_image_ids.append(image_id)
            filtered_captions.append(caption)

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, device):
    """
    使用 CLIP 对 captions 进行编码
    """
    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx: idx + bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    return np.concatenate(encoded_captions)


def encode_images(images, image_path, model, feature_extractor, device):
    """
    使用 CLIP 对 images 进行编码
    """
    image_ids = [i["imgid"] for i in images]
    bs = 1
    image_features = []

    for idx in tqdm(range(0, len(images), bs)):  
        image_input = [
            feature_extractor(Image.open(os.path.join(image_path, i["file_name"])))
            for i in images[idx: idx + bs]
        ]
        with torch.no_grad():
            image_features.append(
                model.encode_image(torch.tensor(np.stack(image_input)).to(device))
                .cpu()
                .numpy()
            )

    return image_ids, np.concatenate(image_features)


def get_nns(captions, images, k=15):
    """
    计算最近邻 captions
    """
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)

    D, I = index.search(xq, k)
    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """
    过滤掉自身 caption, 保留 4 个最近邻 captions
    """
    retrieved_captions = {}

    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 4:
                break
        assert len(good_nns) == 4
        retrieved_captions[image_id] = good_nns

    return retrieved_captions


def main():
    dataset_path = "data/LEVIR_CC/LevirCCcaptions_v1.json"
    image_path = "data/LEVIR_CC/images"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # 1. 筛选 changeflag=1 的 captions
    changed_captions, caption_ids = filter_changed_captions(dataset_path)

    # 2. 过滤 captions
    xb_image_ids, filtered_captions = filter_captions(changed_captions, caption_ids)

    # 3. 编码 captions
    print("Encoding captions...")
    encoded_captions = encode_captions(filtered_captions, clip_model, device)

    # 4. 编码图像
    print("Encoding images...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    images = [item for item in dataset["images"] if item["changeflag"] == 1]

    image_ids, encoded_images = encode_images(images, image_path, clip_model, preprocess, device)

    # 5. 计算最近邻 captions
    print("Retrieving neighbors...")
    index, nns = get_nns(encoded_captions, encoded_images)

    # 6. 过滤最近邻 captions
    retrieved_caps = filter_nns(nns, xb_image_ids, filtered_captions, image_ids)

    # 7. 保存结果
    output_path = "data/LEVIR_CC/retrieved_captions.json"
    with open(output_path, "w") as f:
        json.dump(retrieved_caps, f, indent=2)

    print(f"Saved retrieved captions to {output_path}")


if __name__ == "__main__":
    main()
