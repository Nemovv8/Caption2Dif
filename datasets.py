import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import GPT2Tokenizer
from torchvision import transforms
import pickle

from PIL import Image
import clip
import h5py
import json
import random
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, clip_model_type, dataset_name, data_folder, data_name, split, prefix_length, gpt2_type="gpt2", transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        self.split = split
        self.dataset_name = dataset_name
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        path = os.path.join(data_folder, 'TRAIN' + '_retrived_caps' + '.json')#临时改动
        if split=='TRAIN':
            with open(path) as f:
                retrived_caps = json.load(f)
        clip_model, self.preprocess = clip.load(clip_model_type, device='cpu', jit=False)

        #
        clip_model_name = clip_model_type.replace('/', '_')
        data_path = os.path.join('data/LEVIR_CC/v1', split +'_'+ data_name + '.pkl')

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)  #dict_keys(['images', 'captions', 'caplens'])
            print(all_data)

        self.imgs = all_data['images']  #6815,每个位置存储 两张图片和一个changeflag，里面图片的大小为分别为[256,256,3], self.imgs[1]['ori_img'][0]可以拿到第一张图片

        # Load encoded captions (completely into memory)
        # with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        captions = all_data['captions']  #6815*5

        # Load caption lengths (completely into memory)
        self.caption_lens = all_data['caplens']  #记录captions的长度
        self.caplens=[]
        # FIXME:
        self.prefix_length = prefix_length  #104
        gpt2_type = 'gpt2'
        #gpt2_type = 'D:\Personal\Desktop\\aaai\PromptCC\gpt2'
        # gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        captions_tokens = []
        self.caption2embedding = []
        if self.split=='TRAIN':
            random_key = random.choices(list(retrived_caps.keys()), k=1)
            k = len(retrived_caps[random_key[0]])
            self.cpi = 5 + k
            for i in range(len(captions)):
                captions_tokens.append(torch.tensor(tokenizer.encode(captions[i]), dtype=torch.int64))  #使用gpt2,把captions里的语句转化成对应的编码，长度与caption一致('the scene is the same as before.'=>tensor([1169, 3715,  318,  262,  976,  355,  878,   13]))
                self.caplens.append(self.caption_lens[i])
                if (i+1)%5==0:
                    index=str(int(i/5))
                    if index in retrived_caps:
                        for caps in retrived_caps[index]:
                            captions_tokens.append(torch.tensor(tokenizer.encode(caps[0]), dtype=torch.int64))
                            self.caplens.append(caps[1])
                    else:
                        for _ in range(k):
                            captions_tokens.append(torch.tensor(tokenizer.encode('the scene is the same as before.'), dtype=torch.int64))
                            self.caplens.append(7)
        else:
            self.cpi = 5
            for caption in captions:
                captions_tokens.append(torch.tensor(tokenizer.encode(caption), dtype=torch.int64))
            self.caplens=self.caption_lens
            # self.caption2embedding.append(caption["clip_embedding"])
            # max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        self.max_seq_len = 50
        self.captions = captions_tokens

        # Total number of datapoints
        self.dataset_size = int(len(self.captions) / 1)

    def pad_tokens(self, item: int):
        tokens = self.captions[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            # self.captions[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            # self.captions[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_dict = (self.imgs[i // self.cpi])  # torch.FloatTensor

        if self.dataset_name == 'LEVIR_CC':
            ori_img_list = img_dict['ori_img']
            A = self.preprocess(Image.fromarray(ori_img_list[0])).unsqueeze(0)  #[1,3,224,224]
            B = self.preprocess(Image.fromarray(ori_img_list[1])).unsqueeze(0)
            ori_img = (torch.cat([A, B], dim=0))  #[2,3,224,224]
            changeflag = img_dict['changeflag']

        # FIXME:
        tokens, mask = self.pad_tokens(i)

        caption = torch.LongTensor(tokens)#torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        # changeflag = torch.LongTensor(changeflag)

        if self.split == 'TRAIN':
            # if changeflag==1:
            #     print(changeflag)
            return ori_img, changeflag, caption, mask, caplen
            # ori_img[b,3,224,224]#
            # changeflag0/1
            # captions里已经是对应的编码了，并被统一到了一个一样的长度，不足部分填0，多余部分直接截掉
            # mask是caption的对应，有值的地方填1，被填充了的地方填0
            # caplen是caption的长度
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            # caption_setlist = self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]
            caption_setlist = []
            for k in range((i // self.cpi) * self.cpi,((i // self.cpi) * self.cpi) + self.cpi):
                one_caption, _ = self.pad_tokens(k)
                caption_setlist.append(one_caption.tolist())
            all_captions = torch.LongTensor(caption_setlist)
            return ori_img, changeflag, caption, mask, caplen, all_captions

    def __len__(self):
        return self.dataset_size


