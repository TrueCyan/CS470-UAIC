from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
import torch 
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 
from train import SPECIAL_TOKENS_DICT
from torch.optim import AdamW
from transformers import BertConfig

#  ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_cap(img_feature, model, tokenizer): 
    """Insertion-based 캡션 생성 (훈련 방식과 동일)"""
    
    # 토큰 ID 가져오기
    bos_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_id = tokenizer.convert_tokens_to_ids("[EOS]")
    none_id = tokenizer.convert_tokens_to_ids("[NONE]")
    img_id = tokenizer.convert_tokens_to_ids("[IMG]")
    txt_id = tokenizer.convert_tokens_to_ids("[TXT]")
    
    print(f"토큰 ID: BOS={bos_id}, EOS={eos_id}, NONE={none_id}, IMG={img_id}, TXT={txt_id}")
    
    # [BOS]로 시작
    current_sequence = [bos_id]
    
    max_iterations = 10  # 최대 반복 횟수
    
    with torch.no_grad():
        for iteration in range(max_iterations):
            print(f"\n--- 반복 {iteration+1} ---")
            current_tokens = [tokenizer.convert_ids_to_tokens(id) for id in current_sequence]
            print(f"현재 시퀀스: {' '.join(current_tokens)}")
            
            # 1. 이미지 임베딩 계산 (훈련 시와 동일)
            img_embs = model.image_ff(img_feature)  # (num_regions, hidden_size)
            
            # 2. 텍스트 임베딩 계산
            input_ids_tensor = torch.tensor(current_sequence, dtype=torch.long, device=device)
            text_embs = model.transformer.embeddings.word_embeddings(input_ids_tensor)
            
            # 3. 전체 임베딩 결합 (이미지 + 텍스트)
            combined_embs = torch.cat([img_embs, text_embs], dim=0)
            
            # 4. 토큰 타입 ID 생성 (훈련 시와 동일)
            img_type_ids = [img_id] * img_embs.size(0)
            txt_type_ids = [txt_id] * text_embs.size(0)
            token_type_ids = img_type_ids + txt_type_ids
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
            
            # 5. 토큰 타입 임베딩 추가
            token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids)
            combined_embs = combined_embs + token_type_embs
            
            # 6. 배치 차원 추가
            combined_embs = combined_embs.unsqueeze(0)
            
            # 7. 모델 순전파
            outputs = model(combined_embs)
            logits = outputs[0]  # (1, seq_len, vocab_size)
            
            # 8. 텍스트 부분의 로짓만 추출 (이미지 부분 제외)
            text_start_idx = img_embs.size(0)
            text_logits = logits[0, text_start_idx:, :]  # (text_len, vocab_size)
            
            # 9. 각 위치에서 다음에 삽입할 토큰 예측
            predicted_tokens = torch.argmax(text_logits, dim=-1)  # (text_len,)
            predicted_token_strs = [tokenizer.convert_ids_to_tokens(id.item()) for id in predicted_tokens]
            
            print(f"예측된 토큰들: {predicted_token_strs}")
            
            # 디버깅: 상위 5개 토큰의 확률 분포 확인
            print("=== 출력 분포 분석 ===")
            for pos in range(text_logits.size(0)):
                probs = torch.softmax(text_logits[pos], dim=-1)
                top_probs, top_indices = torch.topk(probs, 10)
                top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
                print(f"위치 {pos}: {list(zip(top_tokens, [f'{p:.4f}' for p in top_probs.tolist()]))}")
            
            # NONE 토큰의 확률이 너무 높은지 확인
            none_prob = torch.softmax(text_logits[0], dim=-1)[none_id].item()
            print(f"[NONE] 토큰 확률: {none_prob:.4f}")
            
            # 특수 토큰들의 ID와 확률 확인
            special_tokens_check = {
                '[NONE]': none_id,
                '[BOS]': bos_id, 
                '[EOS]': eos_id,
                '[IMG]': img_id,
                '[TXT]': txt_id
            }
            
            print("특수 토큰 확률:")
            for token_name, token_id in special_tokens_check.items():
                if token_id < text_logits.size(-1):  # vocab_size 범위 내인지 확인
                    prob = torch.softmax(text_logits[0], dim=-1)[token_id].item()
                    print(f"  {token_name} (ID: {token_id}): {prob:.4f}")
            
            # 일반 단어들 중 상위 확률 토큰들
            print("일반 토큰 중 상위 10개:")
            probs = torch.softmax(text_logits[0], dim=-1)
            # 특수 토큰들을 제외하고 확률 계산
            special_ids = set(special_tokens_check.values())
            for i, token_id in enumerate(torch.argsort(probs, descending=True)):
                if token_id.item() not in special_ids:
                    token = tokenizer.convert_ids_to_tokens(token_id.item())
                    print(f"  {token} (ID: {token_id.item()}): {probs[token_id].item():.4f}")
                    if i >= 9:  # 상위 10개만
                        break
            
            # 10. sequence_stage_combine 적용
            new_sequence = sequence_stage_combine(current_sequence, predicted_token_strs, tokenizer)
            
            # 11. 변화가 없거나 EOS 토큰이 나오면 종료
            if new_sequence == current_sequence:
                print("더 이상 변화가 없어 종료")
                break
            
            if eos_id in [tokenizer.convert_tokens_to_ids(token) for token in predicted_token_strs]:
                print("EOS 토큰으로 종료")
                break
                
            current_sequence = new_sequence
    
    # BOS 제거하고 최종 결과 반환
    final_tokens = [tokenizer.convert_ids_to_tokens(id) for id in current_sequence[1:]]  # BOS 제거
    final_tokens = [token for token in final_tokens if token not in ['[BOS]', '[EOS]', '[NONE]']]
    
    print(f"최종 생성된 캡션: {' '.join(final_tokens)}")
    return final_tokens


def sequence_stage_combine(input_ids, output_tokens, tokenizer):
    """훈련 시와 동일한 시퀀스 결합 방식"""
    new_sequence = []
    
    for i, input_id in enumerate(input_ids):
        new_sequence.append(input_id)
        
        # 해당 위치에 예측된 토큰이 있고 [NONE]이 아니면 추가
        if i < len(output_tokens) and output_tokens[i] != '[NONE]':
            new_token_id = tokenizer.convert_tokens_to_ids(output_tokens[i])
            new_sequence.append(new_token_id)
    
    return new_sequence


def eval():
    ckpt_path = 'ckpt'
    data_path = 'data'

    # 토크나이저 로드 (train.py에서 저장된 것 사용)
    print("훈련된 토크나이저를 로드합니다...")
    try:
        tokenizer = BertTokenizer.from_pretrained(ckpt_path, do_lower_case=False)
        print(f"토크나이저 로드됨. vocab 크기: {len(tokenizer.vocab)}")
        print(f"토크나이저 실제 vocab_size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"저장된 토크나이저 로드 실패: {e}")
        print("기본 토크나이저를 사용하고 특수 토큰을 추가합니다...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        print(f"토크나이저 vocab 크기: {len(tokenizer.vocab)}")
        print(f"토크나이저 실제 vocab_size: {tokenizer.vocab_size}")
    
    # 모델 로드
    print("모델을 로드합니다...")
    try:
        model = UAIC.from_pretrained(ckpt_path)
        print(f"기존 모델 로드됨. 모델 vocab 크기: {model.config.vocab_size}")
        
        # vocab 크기 확인 및 조정
        if tokenizer.vocab_size != model.config.vocab_size:
            print(f"vocab 크기 불일치: 토크나이저({tokenizer.vocab_size}) vs 모델({model.config.vocab_size})")
            print("모델 임베딩 크기를 조정합니다...")
            model.resize_token_embeddings(tokenizer.vocab_size)
            print(f"조정 완료. 새로운 모델 vocab 크기: {model.config.vocab_size}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("새로운 모델을 생성합니다...")
        config = BertConfig(vocab_size=tokenizer.vocab_size)
        model = UAIC(config)
        print(f"새 모델 생성됨. vocab 크기: {model.config.vocab_size}")
    
    model = model.to(device)
    model.eval()
    
    # 디버깅: 토크나이저와 모델 정보 확인
    print(f"최종 토크나이저 vocab 크기: {len(tokenizer.vocab)}")
    print(f"최종 토크나이저 vocab_size: {tokenizer.vocab_size}")
    print(f"최종 모델 vocab 크기: {model.config.vocab_size}")
    print(f"모델 hidden_size: {model.config.hidden_size}")
    
    # 특수 토큰 ID 확인
    special_tokens_info = {
        '[BOS]': tokenizer.convert_tokens_to_ids("[BOS]"),
        '[EOS]': tokenizer.convert_tokens_to_ids("[EOS]"),
        '[NONE]': tokenizer.convert_tokens_to_ids("[NONE]"),
        '[IMG]': tokenizer.convert_tokens_to_ids("[IMG]"),
        '[TXT]': tokenizer.convert_tokens_to_ids("[TXT]"),
        '[PAD]': tokenizer.convert_tokens_to_ids("[PAD]"),
        '[UNK]': tokenizer.convert_tokens_to_ids("[UNK]")
    }
    
    print("특수 토큰 ID 매핑:")
    for token, token_id in special_tokens_info.items():
        print(f"  {token}: {token_id}")
        # vocab 범위 확인
        if token_id >= tokenizer.vocab_size:
            print(f"경고: {token} ID {token_id}가 vocab 크기 {tokenizer.vocab_size}를 초과합니다!")
            return
    
    # vocab_size 일치 확인
    if tokenizer.vocab_size == model.config.vocab_size:
        print("토크나이저와 모델의 vocab_size가 일치합니다.")
        print("모든 특수 토큰이 vocab 범위 내에 있습니다.")
    else:
        print(f"경고: 여전히 불일치합니다!")
        return
    
    smooth = SmoothingFunction()

    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    results = []
    
    # 처음 3개만 테스트
    for i, instance in enumerate(val_data[:1]):  # 1개만 테스트
        print(f"\n{'='*60}")
        print(f"이미지 {i+1} 처리 중")
        print(f"{'='*60}")
        print(f"이미지 ID: {instance['image_id']}")
        print(f"참조 캡션: {instance['caption']}")
        
        img_id = str(instance['image_id']) + '_features'
        
        try:
            img_feature = torch.FloatTensor(img_features[img_id]).to(device)
            print(f"이미지 특성 크기: {img_feature.shape}")
            
            candidates = generate_cap(img_feature, model, tokenizer)
            
            if candidates:
                generated_caption = ' '.join(candidates)
                print(f"\n생성된 캡션: {generated_caption}")
                
                # BLEU 점수 계산
                reference_words = instance['caption'].split()
                bleu_score = corpus_bleu([reference_words], [candidates], 
                                       smoothing_function=smooth.method1)
                results.append(bleu_score)
                print(f"BLEU 점수: {bleu_score:.4f}")
            else:
                print("\n생성된 캡션이 없습니다")
                
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if results:
        avg_bleu = sum(results) / len(results)
        print(f"\n{'='*60}")
        print(f"평균 BLEU 점수: {avg_bleu:.4f}")
        print(f"총 처리된 이미지: {len(results)}개")
    else:
        print("처리된 이미지가 없습니다")


if __name__ == "__main__": 
    eval()