
import time
import os
import numpy as np
from imageio import imread
from skimage.transform import resize as imresize
from random import seed, choice, sample
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse, random
def process_captions(flat_captions):
    processed_captions = []
    caplens = []

    for caption in flat_captions:
        # 确保每个描述以句号结尾
        if not caption.strip().endswith('.'):
            caption += '.'
        
        # 添加到处理后的列表中
        processed_captions.append(caption)
        
        # 计算描述的长度（这里假设以空格分隔的单词数为长度）
        words = caption.split()
        caplen = len(words)
        caplens.append(caplen)

    return processed_captions, caplens

# 调用
 # 打印第二个键值对


def create_input_files(args,dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    template_path = "template.txt"  # 替换为你的模板文件路径
    with open(template_path, "r") as file:
        template = file.read().strip() + " "  # 加载并去掉首尾空格

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_changeflag = []
    train_image_id = []

    val_image_paths = []
    val_image_captions = []
    val_image_changeflag = []
    val_image_id = []

    test_image_paths = []
    test_image_captions = []
    test_image_changeflag = []
    test_image_id = []

    # word_freq = Counter()  # 创建一个空的Counter类(计数

    train_num=0
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            # word_freq.update(c['tokens'])   # 其中c['tokens']是一个很多单词组成的句子‘列表’
            # if len(c['tokens']) <= max_len:
            captions.append(c['raw'].replace(' .','').replace('.',''))  #把句尾的句号去掉
            # print('captions', captions)
            # exit()
        if len(captions) == 0:
            continue

        if dataset == 'LEVIR_CC':
            path1 = os.path.join(image_folder, img['split'], 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'B', img['filename'])
            path = [path1, path2]
            changeflag = img['changeflag']
            image_id = img['imgid']

        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_changeflag.append(changeflag)
            train_image_id.append(image_id)
            # train_num = train_num+1
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_changeflag.append(changeflag)
            val_image_id.append(image_id)


        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_changeflag.append(changeflag)
            test_image_id.append(image_id)
            # print(f'TEST DATA - imgid:{image_id},changeflage : {changeflag}')


    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)


    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img'

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    device = torch.device('cuda:0')
    # print('train_image_captions', train_image_captions)
    test_caption = 'data/LEVIR_CC/v1/TEST_retrived_caps_v2.json'
    train_caption = 'data/LEVIR_CC/v1/TRAIN_retrived_caps_v2.json'
    val_caption = 'data/LEVIR_CC/v1/VAL_retrived_caps_v2.json'
    for impaths, imcaps, imchangeflag, imgid, caption_path, split, in [
                                   (test_image_paths, test_image_captions, test_image_changeflag, test_image_id, test_caption, 'TEST'),
                                   (train_image_paths, train_image_captions, train_image_changeflag, train_image_id, train_caption, 'TRAIN'),
                                    (val_image_paths, val_image_captions, val_image_changeflag, val_image_id, val_caption, 'VAL')]:
        # print('imcaps', imcaps)
        out_path = os.path.join(output_folder, split +'_' + base_filename + '.pkl')
        caps_path = caption_path
        feature_list = []
        enc_captions = []
        enco_captions = []
        caplens = []
        
        retrieved_caps = json.load(open(caps_path))
        counter = 0
        n = 10
        # print('retrieved_caps', type(retrieved_caps))
        # print('retrieved_caps', retrieved_caps)


        for i, path in enumerate(tqdm(impaths)):
            # if len(imchangeflag) <= i:
            #     print(f"Error: imchangeflag is empty or out of index at i={i}")
            #     continue  # 避免报错
            # print(f"Processing index {i}, imgid={imgid[i]}, changeflag={imchangeflag[i]}")
            imgid[i] = str(imgid[i])
            filled_templates = []
            counter += 1
            # print('impaths[i], i, imgid[i] ==============', impaths[i], i, imgid[i])

            # print('imcaps[i]', imcaps[i])
            # if 'the scene is the same as before' in imcaps[i] and split == 'TRAIN':
            #     continue
            # Sample captions
            #处理captions：如果captions不足5个 用已有的填充满五个
            #           如果captions够5个 看是不是train数据集。
            #           如果不是train数据集 或者是train数据集但图片发生了改动 不管。
            #           否则（是train数据集切图片没改变 就这一种情况 需要把5个caption全部变成 'the scene is the same as before'）

            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]

            if split == 'TEST':
                # if not imcaps[i]:
                #     print(f'error: imcaps[{i}] is empty')
                #     continue
                # for nochanged image pairs, just use one kind of nochanged captions during the training
                if imchangeflag[i] == 0:
                        # 填充 || 部分
                    infix = "\n\n".join(["the scene is the same as before"] * 4)  # 复制4次，每次换行
                    template_0 = template.replace("||", infix)  # 替换 || 占位符

                    # 填充 ** 部分
                    
                    j = 0
                    while j <= k:  # 确保只取前五个元素
                        # 替换模板中的 ** 为 caption
                        filled_template = template_0.replace("**", imcaps[i][j])
                        filled_templates.append(filled_template)
                        j += 1  # 将填充后的模板加入结果列表
                    # print("Filled Templates:\n", filled_templates)
                    enc_captions.append(filled_templates)


                    # 打印结果
                    # print("enc_captions:\n", enc_captions)

                else:
                    # print_nth_pair(retrieved_caps, 1) 
                    # 填充||部分
                    print(f"🔍 Checking imgid[{i}] = {imgid[i]}")
                    print(f"🔍 retrieved_caps keys (first 10): {list(retrieved_caps.keys())[:10]}")
                    if imgid[i] in retrieved_caps:
                        value_of_retrieved_caps = retrieved_caps[imgid[i]]  # 获取对应的 value
                        caption_of_retrieved_caps = [lst[0] for lst in value_of_retrieved_caps]
                        merged_captions = "\n".join(caption_of_retrieved_caps)
                        template_1 = template.replace("||", merged_captions)  # 替换 || 占位符
                        # print("Filled Template:\n", template_1)
                    else:
                        print(f"❌ imgid[{i}] = {imgid[i]} NOT found in retrieved_caps!")

                    # 填充 ** 部分
                    filled_templates = []
                    j = 0
                    while j <= k:  # 确保只取前五个元素
                        # 替换模板中的 ** 为 caption
                        filled_template = template_1.replace("**", imcaps[i][j])
                        filled_templates.append(filled_template)
                        j += 1  # 将填充后的模板加入结果列表
                    enc_captions.append(filled_templates)
                    # 打印结果测试
                    # print("Filled Template:\n", filled_template)
                    # print("enc_captions:\n", enc_captions)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            if dataset =='LEVIR_CC':
                ori_img_A = io.imread(impaths[i][0])
                ori_img_B = io.imread(impaths[i][1])
                images = {'ori_img': [ori_img_A, ori_img_B], 'changeflag': imchangeflag[i]}
            else:
                print("Error")

            feature_list.append(images)



        # Sanity check
        # assert len(feature_list) * captions_per_image == len(enc_captions) == len(caplens)
        #把图片对（里面包含两个图片和是否改变的flag）,加上了句号的captions，captions句子的长度 写进文件中
            # print("enco_captions:\n", enco_captions)
        flat_captions = [item for sublist in enc_captions for item in sublist]
        processed_captions, caplens = process_captions(flat_captions)

        print('flat_captions', flat_captions)
        print('caplens', caplens)
        with open(out_path, 'wb') as f:
            pickle.dump({"images": feature_list, "captions": processed_captions, 'caplens': caplens}, f)

if __name__ == '__main__':

    k = 4#num of relative caps
    # print('create_input_files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()

    create_input_files(args, dataset='LEVIR_CC',
                       karpathy_json_path=r'./data/LEVIR_CC/LevirCCcaptions_v1.json',
                       image_folder=r'./data/LEVIR_CC/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r'./data/LEVIR_CC/v2',
                       max_len=50)

    # print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))