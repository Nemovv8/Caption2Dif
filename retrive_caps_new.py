import json
from tqdm import tqdm
from transformers import AutoTokenizer
#import clip
#import torch
import faiss
#import os
import numpy as np
#from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import GPT2Tokenizer
from torchvision import transforms
import pickle

from PIL import Image
import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""
    coco_data_path='data/LEVIR_CC/LevirCCcaptions_v1.json'#not coco_data
    annotations = json.load(open(coco_data_path))["images"]
    images = []
    captions = []
    for item in annotations:
        if item["split"] == "train":
            for sentence in item["sentences"]:
                captions.append(
                    {
                        "image_id": item["imgid"],
                        "caption": " ".join(sentence["tokens"]),
                    }
                )
        images.append(
            {"image_id": item["imgid"], "file_name": item["filename"].split("_")[-1]}
        )

    return images, captions


def filter_captions(data,ids):

    # import os

    # if os.path.exists("filter_captions_image_ids.npy") and os.path.exists(
    #     "filter_captions_captions.npy"
    # ):
    #     filtered_image_ids = np.load("filter_captions_image_ids.npy")
    #     filtered_captions = np.load("filter_captions_captions.npy")
    #     return filtered_image_ids.tolist(), filtered_captions.tolist()

    decoder_name = "gpt2"
    #tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    bs=256

    #image_ids = [d["image_id"] for d in data]
    #caps = [d["caption"] for d in data]
    caps = data
    image_ids = ids
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(
            caps[idx : idx + bs], return_tensors="np", padding=True
        )["input_ids"].tolist()

    filtered_image_ids, filtered_captions = [], []

    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 50:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    # np.save("filter_captions_image_ids", filtered_image_ids)
    # np.save("filter_captions_captions", filtered_captions)

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, device):

    # import os

    # if os.path.exists("encode_captions.npy"):
    #     encode_captions = np.load("encode_captions.npy")
    #     return encode_captions

    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx : idx + bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    # np.save("encode_captions", encode_captions)

    return encoded_captions


def encode_images(images, image_path, model, feature_extractor, device):

    # import os

    # if os.path.exists("encode_img_ids.npy") and os.path.exists(
    #     "encode_img_features.npy"
    # ):
    #     image_ids = np.load("encode_img_ids.npy").tolist()
    #     image_features = np.load("encode_img_features.npy")
    #     return image_ids, image_features

    image_ids = [i["image_id"] for i in images]

    bs = 1
    image_features = []
    #feature_extractor就是把图片reshape一下之后再标准化
    for idx in tqdm(range(0, len(images), bs)):
        image_input = [
            feature_extractor(Image.open(os.path.join(image_path, i["file_name"])))
            for i in images[idx : idx + bs]
        ]
        with torch.no_grad():
            image_features.append(
                model.encode_image(torch.tensor(np.stack(image_input)).to(device))
                .cpu()
                .numpy()
            )

    image_features = np.concatenate(image_features)

    # np.save("encode_img_ids", np.array(image_ids))
    # np.save("encode_img_features", image_features)

    return image_ids, image_features


def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    # import os

    # if os.path.exists("get_nns_D.npy") and os.path.exists("get_nns_I.npy"):
    #     D = np.load("get_nns_D.npy")
    #     I = np.load("get_nns_I.npy")
    #     return index, I
    D, I = index.search(xq, k)   #index是xb（部分caption）的索引。返回xb中 查询向量xq（图片）的 k个xb最相似结果的 距离 和 下标

    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids, caplens):
    """We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:  #这个函数应该就是为了去除自己而存在的，因为自己的文字不应该出现在neighbours里面
                continue
            # good_nns.append(captions[nn])
            # newly added by rfdcomputer
            good_nns.append((captions[nn], caplens[nn], xb_image_ids[nn]))
            if len(good_nns) == 4:
                break
        assert len(good_nns) == 4
        retrieved_captions[image_id] = good_nns
    return retrieved_captions


def main():
    data_folder = './data/LEVIR_CC/v1'
    split = 'TRAIN'
    data_name = 'LEVIR_CC_5_cap_per_img'
    clip_model_type = 'ViT-B/32'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(data_folder, split +'_'+ data_name + '.pkl')

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)  # dict_keys(['images', 'captions', 'caplens'])
    imgs = all_data['images']
    captions = all_data['captions']
    caplens = all_data['caplens']
    difs = []
    #flags = []
    image_ids=[]
    caption_ids = []
    changed_captions = []
    changed_captions_lens = []
    indexi=0
    #imgs = imgs[:100]
    #captions = captions[:500]
    #encode_imgs:
    # for image in imgs:

    #     if image['changeflag']==1:
    #         A = preprocess(Image.fromarray(image['ori_img'][0])).unsqueeze(0)  # [1,3,224,224]
    #         B = preprocess(Image.fromarray(image['ori_img'][1])).unsqueeze(0)
    #         with torch.no_grad():
    #             clip_emb_A, img_feat_A = clip_model.encode_image(A.to(device))  # [1,512]，[1,7*7,768]
    #             clip_emb_B, img_feat_B = clip_model.encode_image(B.to(device))
    #             dif = (clip_emb_B-clip_emb_A).cpu().numpy()
    #             difs.append(dif)
    #             image_ids.append(indexi)
    #             for i in range(5):
    #                 if indexi*5 + i < len(captions):
    #                     changed_captions.append(captions[indexi*5+i])
    #                     changed_captions_lens.append(caplens[indexi*5+i])
    #                     caption_ids.append(indexi)
    #     indexi = indexi+1
        #break
    for image in imgs:
        if image['changeflag'] == 1:
            A = preprocess(Image.fromarray(image['ori_img'][0])).unsqueeze(0)
            B = preprocess(Image.fromarray(image['ori_img'][1])).unsqueeze(0)
            with torch.no_grad():
                clip_emb_A, img_feat_A = clip_model.encode_image(A.to(device))
                clip_emb_B, img_feat_B = clip_model.encode_image(B.to(device))
                dif = (clip_emb_B - clip_emb_A).cpu().numpy()
                difs.append(dif)
                image_ids.append(image['imgid'])  # 改成实际的 imgid
            for i in range(5):
                if indexi * 5 + i < len(captions):
                    changed_captions.append(captions[indexi * 5 + i])
                    changed_captions_lens.append(caplens[indexi * 5 + i])
                    caption_ids.append(image['imgid'])  # 也用实际的 imgid
        indexi += 1


    difs=np.concatenate(difs)
    print("Filtering captions")
    xb_image_ids, changed_captions = filter_captions(changed_captions,caption_ids)  #筛选出部分的captions和对应的image_id，部分的caption指encode的时候长度小于25的,此时captions里还都是字符

    print("Encoding captions")
    encoded_captions = encode_captions(changed_captions, clip_model, device)  # 把上一步经过筛选后的部分captions进行encode得到encoded_captions
    #先把caption里的字符变成对应的数字，其余不足77的地方填充为0，得到[b,77]。然后，使用model（就是clip）里的encode_text方法（这个方法先把输入变成[b,77,1024]，再reshape之后过一个transformer，再reshape回来，最后返回的x的shape是[b,1024]

    #print("Encoding images")
    #xq_image_ids, encoded_images = encode_images(
    #    images, image_path, clip_model, feature_extractor, device
    #)  #xq_image_ids就是所有的image的id，encode_image是所有的图片
    #这里是输入images经过feature_extractor之后得到（b,3,448,448）的tensor，再通过clip_model.encode_image变成[b,1024]->encoded_images
    #xq_image_ids = [i["image_id"] for i in images]

    print("Retrieving neighbors")
    index, nns = get_nns(encoded_captions, difs)   #对captions建立索引，对于每张图片，找出captions里最接近它的k=15个captions
    retrieved_caps = filter_nns(nns, xb_image_ids, changed_captions, image_ids,changed_captions_lens)  #从15个caption中，除去那些本来就是自己的captions，然后从其他的不是自己的ccaptions中找7个最近的captions
    #for ii in range(nns.shape[0]):
    #    goodnns = []
    #    for jj in range(nns.shape[1]):
    #        if xb_image_ids[nns[ii][jj]]!=xb_image_ids[ii]

    #print("Writing files")
    #faiss.write_index(index, "datastore/coco_index")
    #json.dump(captions, open("datastore/coco_index_captions.json", "w"), indent=2)
    path =  os.path.join(data_folder, split + '_retrived_caps_v2.json')
    json.dump(
        retrieved_caps, open(path, "w"), indent=2
    )
    print(path)


if __name__ == "__main__":
    main()
