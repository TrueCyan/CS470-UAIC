from transformers import * 
import torch.nn as nn
import torch 
from torch.nn import CrossEntropyLoss 
import os 


class UAIC(BertPreTrainedModel):
    def __init__(self, config):
        super(UAIC, self).__init__(config)
        self.transformer = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.image_ff = nn.Linear(2048, config.hidden_size)
        self.image_inverse_ff = nn.Linear(config.hidden_size, 2048)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.embeddings.word_embeddings)

    def get_input_embeddings(self):
        """입력 임베딩 레이어 반환"""
        return self.transformer.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """입력 임베딩 레이어 설정"""
        self.transformer.embeddings.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        """출력 임베딩 레이어 반환"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """출력 임베딩 레이어 설정"""
        self.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        """토큰 임베딩 크기 조정"""
        if new_num_tokens is None:
            return self.get_input_embeddings()
        
        old_embeddings = self.get_input_embeddings()
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # 새로운 임베딩 레이어 생성
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # 기존 가중치 복사
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        # 입력 임베딩 업데이트
        self.set_input_embeddings(new_embeddings)
        
        # 출력 레이어도 업데이트
        old_lm_head = self.get_output_embeddings()
        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=old_lm_head.bias is not None)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # 기존 가중치 복사
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        
        self.set_output_embeddings(new_lm_head)
        
        # config 업데이트
        self.config.vocab_size = new_num_tokens
        
        # 가중치 다시 tie
        self.tie_weights()
        
        return self.get_input_embeddings()

    def forward(self, input_embs, labels=None):
        transformer_outputs = self.transformer(inputs_embeds=input_embs)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            # 배치 처리를 위해 squeeze 제거
            batch_size = lm_logits.size(0)
            seq_length = lm_logits.size(1)
            vocab_size = lm_logits.size(2)
            
            # 배치 크기와 시퀀스 길이 불일치 처리
            if seq_length != labels.size(1):
                min_len = min(seq_length, labels.size(1))
                lm_logits = lm_logits[:, :min_len, :]
                labels = labels[:, :min_len]
            
            # CrossEntropyLoss는 (batch_size * seq_length, vocab_size) 형태와
            # (batch_size * seq_length) 형태의 타겟을 예상함
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_logits_flat = lm_logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            loss = loss_fct(lm_logits_flat, labels_flat)
            outputs = (loss,) + outputs
        
        return outputs 


if __name__ == "__main__":
    model_path = 'ckpt'
    
    # 실제 vocab.txt 파일 라인 수에 맞춰 설정
    with open(os.path.join(model_path, 'vocab.txt'), 'r', encoding='utf-8') as f:
        actual_vocab_size = sum(1 for _ in f)
    
    print(f"실제 vocab.txt 크기: {actual_vocab_size}")
    
    configration = BertConfig(vocab_size=actual_vocab_size)
    model = UAIC(configration)
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    model.config.to_json_file(os.path.join(model_path, 'config.json'))
    
    # 테스트
    input_embs = torch.rand(5, 768).view(1, -1, 768)
    output = model(input_embs)
    print(f"출력 크기: {output[0].size()}")
    print(f"모델 vocab_size: {model.config.vocab_size}")
