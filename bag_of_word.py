from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch 
from dataset import BagWordsDataset, worker_init_fn
import torch.optim 
from utilis import AverageMeter
import os 
import json 
import h5py 
from torch.nn.functional import softmax, pad
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import numpy
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


# Uncertainty-aware image-conditioned bag-of-words
class BagofWords(BertPreTrainedModel):
    def __init__(self, config):
        super(BagofWords, self).__init__(config)
        self.feature_embd = nn.Linear(2048, config.hidden_size)
        self.transformer = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def acc_compute(self, output, labels):
        # Sigmoid 적용하여 확률로 변환 
        probs = torch.sigmoid(output)
        # 임계값 0.5로 이진 예측 생성
        predictions = (probs >= 0.5).float()
        # 정확한 예측 계산
        correct = (predictions == labels).float()
        # 정확도 반환
        return correct.mean().item()

    
    def forward(self, img_embs, labels=None):
        # 패딩된 이미지 데이터를 처리합니다
        batch_size, seq_len, feat_dim = img_embs.size()
        
        # 임베딩 변환
        img_embs = self.feature_embd(img_embs)  # [batch_size, seq_len, hidden_size]
        
        # 트랜스포머에 입력
        transformer_outputs = self.transformer(inputs_embeds=img_embs)
        hidden_states = transformer_outputs[1]  # [batch_size, hidden_size]
        pool_outputs = self.dropout(hidden_states)
        pool_outputs = self.classifier(pool_outputs)  # [batch_size, vocab_size]

        if labels is None:
            return pool_outputs

        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(pool_outputs, labels)
        acc = self.acc_compute(pool_outputs, labels)

        return loss, acc


# 다양한 크기의 이미지를 처리하기 위한 커스텀 collate 함수
def custom_collate_fn(batch):
    # 배치에서 이미지와 레이블 분리
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 이미지 크기 확인
    max_len = max([img.size(0) for img in imgs])
    
    # 패딩된 이미지 텐서 생성
    padded_imgs = []
    for img in imgs:
        # 현재 이미지와 최대 크기의 차이 계산
        pad_size = max_len - img.size(0)
        
        if pad_size > 0:
            # 패딩 추가 [패드할 차원의 앞, 패드할 차원의 뒤]
            padded_img = pad(img, [0, 0, 0, pad_size], 'constant', 0)
        else:
            padded_img = img
        
        padded_imgs.append(padded_img)
    
    # 텐서로 변환
    imgs_tensor = torch.stack(padded_imgs, dim=0)
    labels_tensor = torch.stack(labels, dim=0)
    
    return imgs_tensor, labels_tensor


# train the image conditioned bag-of-words 
def train():
    epochs = 10 
    model_path = 'model'
    gradient_accumlation_steps = 5 
    batch_size = 32
    num_workers = 4

    tokenizer = BertTokenizer('data/vocab.txt') 
    configuration = BertConfig(vocab_size=tokenizer.vocab.get('[UNK]') + 1, \
                                num_hidden_layers=3, \
                                intermediate_size=2048)
    model = BagofWords(configuration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 옵티마이저 설정 - 학습률 낮춤
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 데이터셋 및 데이터로더 설정
    dataset = BagWordsDataset('data', tokenizer)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        collate_fn=custom_collate_fn
    )
    
    # 학습률 스케줄러 설정 - 정확한 step 계산
    steps_per_epoch = len(train_loader) // gradient_accumlation_steps
    total_steps = steps_per_epoch * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps
    )
    
    # 혼합 정밀도 학습을 위한 scaler 초기화
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # 디렉토리 생성
    os.makedirs(model_path, exist_ok=True)
    
    print(f"훈련 설정: 에포크={epochs}, 배치크기={batch_size}, 그라디언트누적={gradient_accumlation_steps}")
    print(f"에포크당 스텝: {steps_per_epoch}, 총 스텝: {total_steps}")
    
    # 학습 시작
    model.train()
    loss_list = []
    acc_list = []
    
    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        iteration = 1 
        print(f'에포크: {epoch}')
        print(f'데이터셋 크기: {len(dataset)}')
        
        # 에포크 시작 시 그래디언트 초기화
        optimizer.zero_grad()
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if iteration % 1000 == 0:
                print(f'반복: {iteration}')
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 혼합 정밀도 학습 적용
            if use_amp:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    loss, acc = model(imgs, labels)
                    loss = loss / gradient_accumlation_steps
                
                # 그래디언트 계산
                scaler.scale(loss).backward()
                
                # 그래디언트 누적 완료 시에만 업데이트
                if (batch_idx + 1) % gradient_accumlation_steps == 0:
                    # 그래디언트 클리핑
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 파라미터 업데이트
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 학습률 스케줄러 스텝
                    scheduler.step()
                    
                # 마지막 배치 처리 (누적이 완료되지 않은 경우)
                elif batch_idx == len(train_loader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
            else:
                # CPU 학습
                loss, acc = model(imgs, labels)
                loss = loss / gradient_accumlation_steps
                loss.backward()
                
                # 그래디언트 누적 완료 시에만 업데이트
                if (batch_idx + 1) % gradient_accumlation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                # 마지막 배치 처리
                elif batch_idx == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 손실 및 정확도 업데이트
            avg_loss.update(loss.detach().item() * gradient_accumlation_steps)
            avg_acc.update(acc)
            iteration += 1
            
        # 모델 저장
        model_filename = f'epoch{epoch}_acc_{avg_acc.avg:.3f}'.replace('.', '_')
        save_path = os.path.join(model_path, model_filename)
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'loss': avg_loss.avg,
            'accuracy': avg_acc.avg
        }, save_path)
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        
        # 에포크 결과 기록
        loss_list.append(avg_loss.avg)
        acc_list.append(avg_acc.avg)
        
        print(f"에포크 {epoch}: 손실={avg_loss.avg:.4f}, 정확도={avg_acc.avg:.4f}")

    print("최종 손실 목록:", loss_list)
    print("최종 정확도 목록:", acc_list)


# use the bag-of-words model to comput the uncertainty 
# output: json image_id, caption, uncertainty 
def uncertainty_estimation():
    """완전 자동화된 불확실성 추정 함수"""
    print("불확실성 추정 시작")
    
    # 기본 설정
    model_path = 'model'
    data_path = 'data'
    output_path = os.path.join(data_path, 'uncertainty_captions.json')
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 모델 파일 확인
        model_files = [f for f in os.listdir(model_path) if f.startswith('epoch') and '_acc_' in f]
        if not model_files:
            print("훈련된 모델이 없습니다. bag-of-words 모델을 먼저 훈련해주세요.")
            return False
        
        # 정확도가 가장 높은 모델 선택
        def extract_accuracy(filename):
            try:
                return float(filename.split('_acc_')[1])
            except:
                return 0.0
        
        best_model = max(model_files, key=extract_accuracy)
        ckpt_path = os.path.join(model_path, best_model)
        print(f"모델 로드: {best_model}")
        
        # 토크나이저와 모델 로드
        tokenizer = BertTokenizer(os.path.join(data_path, 'vocab.txt'))
        model_config = BertConfig.from_pretrained(model_path)
        model = BagofWords(model_config)
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        print("모델 로드 완료")
        
        unk = tokenizer._convert_token_to_id('[UNK]')
        
        # 캡션 데이터 로드
        caption_path = os.path.join(data_path, 'annotations', 'captions_train2014.json')
        if not os.path.exists(caption_path):
            print("캡션 파일이 없습니다:", caption_path)
            return False
        
        print("캡션 데이터 로드중...")
        with open(caption_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        
        annotations = captions['annotations']  # 전체 데이터 처리
        print(f"{len(annotations)}개 캡션 처리")
        
        # 이미지 특성 파일 확인
        img_features_path = os.path.join(data_path, 'coco_detections.hdf5')
        if not os.path.exists(img_features_path):
            print("이미지 특성 파일이 없습니다:", img_features_path)
            return False
        
        result_list = []
        
        # h5py 파일로 처리
        with h5py.File(img_features_path, 'r') as img_features:
            for i, sample in enumerate(tqdm(annotations, desc="처리 중")):
                try:
                    img_id = str(sample['image_id']) + '_features'
                    
                    if img_id not in img_features:
                        continue
                    
                    # 이미지 특성 로드
                    img_tensor = torch.FloatTensor(numpy.array(img_features[img_id])).unsqueeze(0).to(device)
                    
                    # 모델 추론
                    with torch.no_grad():
                        output = model(img_embs=img_tensor)
                    
                    # 캡션 토큰화
                    caption_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['caption']))
                    
                    # 불확실성 계산
                    uncertainty = []
                    for token_id in caption_tokens:
                        if token_id == unk or token_id >= output.size(-1):
                            uncertainty.append(0.0)
                        else:
                            uncertainty.append(float(output[0, token_id].cpu().item()))
                    
                    # 결과 저장
                    new_sample = sample.copy()
                    new_sample['uncertainty'] = uncertainty
                    result_list.append(new_sample)
                    
                except Exception as e:
                    print(f"샘플 {i} 처리 오류: {e}")
                    continue
        
        # 결과 저장
        result_data = {'annotations': result_list}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"완료! {len(result_list)}개 캡션 처리됨")
        print(f"결과 파일: {output_path}")
        return True
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # train the image conditioned bag-of-words 
    train()

    # uncertainty estimation with trained bag-of-word model 
    uncertainty_estimation()


    #s = 'a cat shsiss dog'
    #s = tokenizer.tokenize(s)
    # print(tokenizer.convert_tokens_to_ids(s))


