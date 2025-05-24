import json 
import torch 
import os 
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import functools
# from torch.profiler import profile, record_function, ProfilerActivity # 프로파일러 import 제거

from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from model import UAIC 
from dataset import UAICDataset 
from torch.optim import AdamW 
from torch.amp import GradScaler, autocast

SPECIAL_TOKENS_DICT = {'bos_token': "[BOS]", 'eos_token': "[EOS]", 'additional_special_tokens': ["[NONE]", "[IMG]", "[TXT]"], 'pad_token': "[PAD]"}
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]

# collate_fn을 전역 범위로 이동하고 pad_token_id를 인자로 받도록 수정
def collate_fn(batch, text_pad_token_id, label_pad_token_id, image_pad_value, type_pad_token_id):
    # Ensure elements from batch are converted to lists for pad_sequence
    img_list, input_txt_list, output_list, token_type_id_list_of_lists = zip(*batch)

    # Pad images
    # Assuming img_list contains tensors of shape (img_seq_len, feature_size)
    imgs_padded = pad_sequence(list(img_list), batch_first=True, padding_value=image_pad_value)

    # Ensure 1D by reshaping for sequence-like tensors
    input_txts_1d = [t.view(-1) for t in input_txt_list]
    outputs_1d = [t.view(-1) for t in output_list] # Problematic line fix
    token_type_ids_1d = [t.view(-1) for t in token_type_id_list_of_lists]

    # Pad input_txt
    input_txts_padded = pad_sequence(input_txts_1d, batch_first=True, padding_value=text_pad_token_id)

    # Pad outputs (labels)
    outputs_padded = pad_sequence(outputs_1d, batch_first=True, padding_value=label_pad_token_id)

    # Pad token_type_ids
    token_type_ids_padded = pad_sequence(token_type_ids_1d, batch_first=True, padding_value=type_pad_token_id)

    return imgs_padded, input_txts_padded, outputs_padded, token_type_ids_padded

def train():
    torch.backends.cudnn.benchmark = True # cudnn benchmark 설정 추가
    model_path = 'ckpt'
    data_path = 'data'
    # ckpt_path = 'ckpt'
    lr = 1e-4
    epochs = 15 
    gradient_accumulation_steps = 2 
    batch_size = 8
    num_workers = 4

    tokenizer = BertTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    # functools.partial을 사용하여 pad_token_id를 collate_fn에 바인딩
    custom_collate_fn = functools.partial(collate_fn,
                                          text_pad_token_id=tokenizer.pad_token_id,
                                          label_pad_token_id=-100, # For CrossEntropyLoss ignore_index
                                          image_pad_value=0.0,    # For float image features
                                          type_pad_token_id=0)    # For token type IDs

    model = UAIC.from_pretrained(model_path)
    # print(len(tokenizer))
    # model.transformer.resize_token_embeddings(10877)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_dataset = UAICDataset(data_path, tokenizer)
    
    # bos, eos, none, img_label, txt_label = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    
    # 각 에폭에서 데이터셋의 절반을 사용하므로, 에폭당 예상 반복 횟수를 계산합니다.
    # DataLoader는 에폭마다 재생성되므로, 길이는 에폭마다 동일합니다 (데이터셋 크기의 절반 / 배치 크기).
    # 여기서는 첫 번째 에폭의 DataLoader 길이를 기준으로 전체 학습 스텝을 계산합니다.
    # 또는, train_dataset 길이의 절반을 기준으로 계산할 수도 있습니다.
    # num_training_steps는 gradient_accumulation_steps를 고려해야 합니다.
    
    # 전체 데이터셋 크기의 절반을 기준으로 에폭당 반복 횟수 계산
    effective_dataset_size_per_epoch = len(train_dataset) // 2
    iterations_per_epoch = (effective_dataset_size_per_epoch + batch_size - 1) // batch_size # 올림 처리
    num_training_steps = epochs * iterations_per_epoch // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    for epoch in range(epochs):
        
        # 매 에폭마다 데이터셋의 절반을 랜덤하게 선택
        dataset_size = len(train_dataset)
        # indices = list(range(dataset_size)) # 사용되지 않으므로 주석 처리 또는 삭제
        split = dataset_size // 2
        torch.manual_seed(epoch) # 매 에폭 다른 샘플을 위해 시드 설정 (선택적)
        
        # 실제 사용할 인덱스 섞기
        shuffled_indices = torch.randperm(dataset_size).tolist()
        train_indices = shuffled_indices[:split]
        
        train_sampler = SubsetRandomSampler(train_indices)
        
        # shuffle=False로 설정해야 SubsetRandomSampler가 정상 작동합니다.
        # pin_memory=True는 CUDA 사용 시 데이터 로딩 속도를 향상시킬 수 있습니다.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True)

        iteration = 1
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            img, input_txt, output, token_type_ids = batch
            img = img.to(device)
            input_txt = input_txt.to(device)
            output = output.to(device)
            token_type_ids = token_type_ids.to(device)

            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                input_txt_embs = model.transformer.embeddings.word_embeddings(input_txt)
                img_embs = model.image_ff(img)
                
                combined_input_embs = torch.cat([img_embs, input_txt_embs], dim=1)
                
                out = model(combined_input_embs, output)
                loss = out[0]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if iteration % gradient_accumulation_steps == 0: 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step() # 스케줄러 스텝 추가
            
            progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")
            iteration += 1
            
    
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    # tokenizer.save_vocabulary(model_path)
    


if __name__ == "__main__":
    train()

