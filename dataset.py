import torch 
from torch.utils.data import Dataset
import json
import os 
import h5py
from transformers import BertTokenizer
import numpy
import random

# SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]

# 각 워커에서 h5py 파일을 열기 위한 함수
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # 각 워커는 자신만의 h5py 파일 객체를 열어야 함
    if hasattr(dataset, 'img_features_file'):
        dataset.img_features = h5py.File(dataset.img_features_file, 'r')

# Dataset for uncertainty measurer: bag-of-words 
class BagWordsDataset(Dataset): 

    "output: image region, one-hot vocabulary vector"
    def __init__(self, data_path, tokenizer, cache_size=1000):
        self.img_features_file = os.path.join(data_path, 'coco_detections.hdf5')
        # 메인 프로세스에서만 파일을 열고, 워커는 worker_init_fn에서 열게 됨
        self.img_features = None
        caption_path = os.path.join(data_path, 'annotations')
        train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
        with open(train_caption_path, 'r', encoding='utf-8') as j:
            self.captions = json.load(j) 
        self.tokenizer = tokenizer
        
        # 캐싱을 위한 설정
        self.cache_size = cache_size
        self.img_cache = {}
        self.label_cache = {}
        
        # 메인 프로세스에서는 프리로드 수행
        if cache_size > 0:
            # 임시로 파일 열기
            temp_file = h5py.File(self.img_features_file, 'r')
            self._preload_frequent_items(temp_file)
            temp_file.close()
    
    def _preload_frequent_items(self, h5_file):
        print("자주 사용되는 이미지 미리 캐싱 중...")
        # 이미지 ID 빈도 카운트
        img_id_counts = {}
        for item in self.captions['annotations']:
            img_id = str(item['image_id']) + '_features'
            img_id_counts[img_id] = img_id_counts.get(img_id, 0) + 1
        
        # 빈도 기준 상위 cache_size개 이미지 선택
        most_frequent = sorted(img_id_counts.items(), key=lambda x: x[1], reverse=True)[:self.cache_size]
        
        # 미리 캐싱
        for img_id, _ in most_frequent:
            if img_id in h5_file:
                self.img_cache[img_id] = torch.FloatTensor(numpy.array(h5_file[img_id]))
        
        print(f"{len(self.img_cache)}개 이미지 캐싱 완료")

    def __getitem__(self, i): 
        # 필요한 경우 파일 열기
        if self.img_features is None:
            self.img_features = h5py.File(self.img_features_file, 'r')
            
        cap_dict = self.captions['annotations'][i]
        img_id = str(cap_dict['image_id']) + '_features'
        
        # 캐시에서 이미지 가져오기 또는 로드
        if i in self.label_cache and img_id in self.img_cache:
            return self.img_cache[img_id], self.label_cache[i]
        
        # 이미지 로드 (캐시에 없는 경우)
        if img_id not in self.img_cache:
            img = torch.FloatTensor(numpy.array(self.img_features[img_id]))
            # 캐시 관리 (메모리 오버플로우 방지)
            if len(self.img_cache) < self.cache_size:
                self.img_cache[img_id] = img
        else:
            img = self.img_cache[img_id]
        
        # 레이블 생성
        caption = cap_dict['caption']
        caption = self.tokenizer.tokenize(caption)
        caption = self.tokenizer.convert_tokens_to_ids(caption) 
        label = torch.zeros(self.tokenizer.vocab.get('[UNK]') + 1)
        for idx in caption:
            label[idx] = 1
            
        # 레이블 캐싱
        if len(self.label_cache) < self.cache_size * 2:  # 레이블은 더 많이 캐싱
            self.label_cache[i] = label
        
        return img, label
    
    def __len__(self):
        return len(self.captions['annotations'])


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)    


# Dataset for uncertainty-aware image captioning 
class UAICDataset(Dataset):

    "output: image region, input txt, output txt"
    def __init__(self, data_path, tokenizer, cache_size=1000):
        self.tokenizer = tokenizer 
        self.img_features_file = os.path.join(data_path, 'coco_detections.hdf5')
        # 메인 프로세스에서만 파일을 열고, 워커는 worker_init_fn에서 열게 됨
        self.img_features = None
        data_pair_path = os.path.join(data_path, 'data_pair.json')
        with open(data_pair_path, 'r', encoding='utf-8') as j:
            self.txt = json.load(j)
        self.img_label = tokenizer._convert_token_to_id('[IMG]')
        self.txt_label = tokenizer._convert_token_to_id('[TXT]')
        
        # 캐싱을 위한 설정
        self.cache_size = cache_size
        self.img_cache = {}
        self.data_cache = {}
        
        # 메인 프로세스에서는 프리로드 수행
        if cache_size > 0:
            # 임시로 파일 열기
            temp_file = h5py.File(self.img_features_file, 'r')
            self._preload_frequent_items(temp_file)
            temp_file.close()
    
    def _preload_frequent_items(self, h5_file):
        print("UAICDataset: 자주 사용되는 이미지 미리 캐싱 중...")
        # 이미지 ID 빈도 카운트
        img_id_counts = {}
        for item in self.txt:
            img_id = str(item['image_id']) + '_features'
            img_id_counts[img_id] = img_id_counts.get(img_id, 0) + 1
        
        # 빈도 기준 상위 cache_size개 이미지 선택
        most_frequent = sorted(img_id_counts.items(), key=lambda x: x[1], reverse=True)[:self.cache_size]
        
        # 미리 캐싱
        for img_id, _ in most_frequent:
            if img_id in h5_file:
                self.img_cache[img_id] = torch.FloatTensor(numpy.array(h5_file[img_id]))
        
        print(f"UAICDataset: {len(self.img_cache)}개 이미지 캐싱 완료")
    
    def __getitem__(self, i):
        # 필요한 경우 파일 열기
        if self.img_features is None:
            self.img_features = h5py.File(self.img_features_file, 'r')
            
        # 캐시에서 데이터 가져오기
        if i in self.data_cache:
            return self.data_cache[i]
            
        cap_dict = self.txt[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        
        # 이미지 로드 (캐시에서 가져오거나 새로 로드)
        if img_id not in self.img_cache:
            img = torch.FloatTensor(numpy.array(self.img_features[img_id]))
            if len(self.img_cache) < self.cache_size:
                self.img_cache[img_id] = img
        else:
            img = self.img_cache[img_id]
            
        place_holder = torch.Tensor([-100] * img.size(0)).long()
        input_ids = torch.Tensor(self.str2id('[BOS]'+ cap_dict['input'])).long()
        output_ids = torch.Tensor(self.str2id(cap_dict['output'])).long()
        output_ids = torch.cat([place_holder, output_ids], dim=0)
        if output_ids.dim() == 1: # 1차원인 경우 차원 추가
            output_ids = output_ids.unsqueeze(0)
        token_type_ids = [self.img_label] * img.size(0) + [self.txt_label] * input_ids.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long()

        # 결과 캐싱
        result = (img, input_ids, output_ids, token_type_ids)
        if len(self.data_cache) < self.cache_size:
            self.data_cache[i] = result
            
        return result

    def str2id(self, sentence, max_length=128):
        # 토큰화와 ID 변환을 한 번에 처리하며 truncation 및 max_length 적용
        encoded_inputs = self.tokenizer(
            sentence,
            max_length=max_length,
            truncation=True,
            padding=False, # 패딩은 collate_fn에서 처리하므로 여기서는 False
            return_attention_mask=False # 어텐션 마스크는 필요 시 생성
        )
        return encoded_inputs['input_ids']

    def __len__(self):
        return len(self.txt)



if __name__ == "__main__":
    path = 'data'
    tokenizer = BertTokenizer('data/vocab.txt')
    #data_set = BagWordsDataset(path, tokenizer)
    #img, label = data_set[0] 
    data_set = UAICDataset(path, tokenizer)
    img, input, output = data_set[0]
    print(input)
    print(output)

