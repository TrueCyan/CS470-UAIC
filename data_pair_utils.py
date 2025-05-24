import json 
import numpy as np 
import os 
import random

# Generate training data pair
def create_data_pair(data_path):
    """
    데이터 페어 생성 함수 - 위치 기반 [NONE] 토큰 삽입
    
    Args:
        data_path: 데이터 경로
    """
    uncertainty_json_path = os.path.join(data_path, 'uncertainty_captions.json')
    
    if not os.path.exists(uncertainty_json_path):
        print("uncertainty_captions.json 파일이 없습니다.")
        print("먼저 bag_of_word.py의 uncertainty_estimation()을 실행해주세요.")
        return False
    
    print("불확실성 캡션 데이터 로드중...")
    with open(uncertainty_json_path, 'r', encoding='utf-8') as j:
        uncertainty_json = json.load(j)
    
    train_data = uncertainty_json['annotations']
    print(f"총 {len(train_data)}개 캡션 처리 시작")
    
    data_pair_list = []
    
    for idx, sample in enumerate(train_data):
        if idx % 10000 == 0:
            print(f"진행률: {idx}/{len(train_data)} ({idx/len(train_data)*100:.1f}%)")
        
        try:
            sample_uncertainty = sample['uncertainty'] 
            caption_uncertainty = sample['caption']
            
            if len(sample_uncertainty) == 0:
                continue
            
            order_list = [0] * len(sample_uncertainty)
            tree_construct(sample_uncertainty, 0, len(order_list)-1, order_list, 0)
            
            max_iter = max(order_list) if order_list else 0
            caption = caption_uncertainty.split()
            
            # 각 반복에 대해 데이터 페어 생성
            for i in range(max_iter):
                tmp_sample = {}
                tmp_sample['image_id'] = sample['image_id']
                
                # 입력 시퀀스 생성: 현재 레벨까지의 모든 단어
                input_tokens = []
                for j in range(len(caption)):
                    if order_list[j] <= i:
                        input_tokens.append(caption[j])
                
                input_txt = ' '.join(input_tokens) + ' ' if input_tokens else ''
                
                # 출력 시퀀스 생성: 위치 기반 [NONE] 토큰 삽입
                # 현재 레벨까지의 모든 단어들을 위치 순서대로 정렬
                current_words = []
                for j in range(len(caption)):
                    if order_list[j] <= i+1:  # 현재까지 + 다음 레벨
                        current_words.append((j, caption[j], order_list[j]))
                
                # 위치 순서대로 정렬
                current_words.sort(key=lambda x: x[0])
                
                # 출력 시퀀스 생성
                output_tokens = []
                for pos, word, level in current_words:
                    if level <= i:  # 입력에 이미 있는 단어
                        output_tokens.append('[NONE]')
                    else:  # level == i+1, 새로 추가되는 단어
                        output_tokens.append(word)
                
                output_txt = ' '.join(output_tokens) + ' '
                
                tmp_sample['input'] = input_txt
                tmp_sample['output'] = output_txt
                
                # 빈 출력 방지
                if not tmp_sample['output'].strip():
                    tmp_sample['output'] = '[NONE] '
                
                data_pair_list.append(tmp_sample)
                
        except Exception as e:
            print(f"샘플 {idx} 처리 중 오류: {e}")
            continue
    
    print(f"\n데이터 생성 완료: 총 {len(data_pair_list):,}개 데이터 페어")
    
    # 결과 저장
    output_path = os.path.join(data_path, 'data_pair.json')
    print(f"저장 중: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_pair_list, f, ensure_ascii=False, indent=2)
    
    print("data_pair.json 생성 완료!")
    
    return True


# create the binary tree structure recursively 
def tree_construct(uncertainty_list, left, right, res, level): 
    if left > right:
        return 
    if left == right:
        res[left] = level 
        return 
    idx = left
    max_value = uncertainty_list[idx]
    current_idx = left + 1  
    while current_idx <= right:
        if uncertainty_list[current_idx] > max_value:
            idx =  current_idx
            max_value = uncertainty_list[current_idx]
        current_idx += 1
    res[idx] = level
    tree_construct(uncertainty_list, left, idx-1, res, level+1)
    tree_construct(uncertainty_list, idx+1, right, res, level+1)
    return 


if __name__ == "__main__":
    data_path = 'data'
    create_data_pair(data_path) 