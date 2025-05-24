import json
import torch
from evaluation import compute_scores
from transformers import AutoTokenizer, AutoModelForCausalLM  # 모델에 맞게 임포트 수정 필요

def evaluate_model(model_path, test_data_path):
    """
    모델을 BLEU, METEOR, ROUGE, CIDEr 메트릭으로 평가합니다.
    
    Args:
        model_path (str): 모델 체크포인트 경로
        test_data_path (str): 테스트 데이터 파일 경로 (JSON 형식)
    
    Returns:
        dict: 각 메트릭별 점수
    """
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터 로드
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Ground Truth와 생성된 텍스트를 저장할 딕셔너리
    gts = {}
    gen = {}
    
    # 모델로 텍스트 생성 및 평가 데이터 준비
    with torch.no_grad():
        for i, item in enumerate(test_data):
            # 여기서는 예시로 입력과 참조 텍스트가 이미 있다고 가정
            input_text = item['input']
            reference = item['reference']
            
            # 입력 텍스트 토크나이징 및 모델 추론
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs.input_ids, 
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 평가를 위한 형식으로 데이터 준비
            gts[i] = [reference]  # 참조 텍스트 (리스트 형태)
            gen[i] = [generated_text]  # 생성된 텍스트 (리스트 형태)
    
    # 모든 메트릭으로 평가 실행
    scores, all_scores = compute_scores(gts, gen)
    
    # 결과 출력
    print("평가 결과:")
    for metric, score in scores.items():
        if metric == "Bleu":
            # BLEU는 여러 값(BLEU-1, BLEU-2, BLEU-3, BLEU-4)을 반환
            for i, s in enumerate(score):
                print(f"  BLEU-{i+1}: {s:.4f}")
        else:
            print(f"  {metric}: {score:.4f}")
    
    return scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--model_path", type=str, required=True, help="모델 체크포인트 경로")
    parser.add_argument("--test_data", type=str, required=True, help="테스트 데이터 파일 경로")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data) 