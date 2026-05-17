"""
Interface de Chat para o modelo GpSRNN-1B.
Permite conversar com o modelo via terminal, gerenciando histórico e estados.

PRÉ-REQUISITOS:
    pip install transformers torch

USO:
    python chat_interface.py --checkpoint caminho/do/modelo.pt
    (Se nenhum checkpoint for passado, usará pesos aleatórios para demonstração).
"""

import torch
import argparse
import sys
import os
from transformers import GPT2Tokenizer

# Importa nossa arquitetura
from gp_srrn_1b import GpSRNNConfig, GpSRNNModel, count_parameters

def load_model(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Carrega o modelo e o tokenizer."""
    print(f"🔧 Configurando dispositivo: {device}")
    
    # Configuração idêntica ao treino
    config = GpSRNNConfig(
        vocab_size=50257,
        d_model=1536,
        n_layers=24,
        n_heads=8,
        dropout=0.0  # Dropout 0 na inferência
    )
    
    print("🏗️  Inicializando arquitetura GpSRNN-1B...")
    model = GpSRNNModel(config)
    model = model.to(device)
    model.eval()  # Modo de avaliação (desativa dropout, etc.)
    
    # Se houver checkpoint, carrega os pesos
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📥 Carregando pesos de: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Lidar com casos onde o checkpoint é um dict {'model_state_dict': ...} ou direto o state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Pesos carregados com sucesso!")
    else:
        print("⚠️  Nenhum checkpoint fornecido. Usando pesos aleatórios.")
        print("   (O modelo irá gerar texto sem sentido até ser treinado)")

    # Contagem de parâmetros
    params = count_parameters(model)
    print(f"📊 Parâmetros totais: {params['total_billions']:.2f} Bilhões")

    # Tokenizer (GPT-2 compatible)
    print("🔤 Carregando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def decode_tokens(tokens, tokenizer):
    """Decodifica uma lista de tokens em string, limpando caracteres estranhos se necessário."""
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

@torch.no_grad()
def chat_loop(model, tokenizer, device, max_context_length=1024, max_new_tokens=256, temperature=0.8, top_p=0.9):
    """Loop principal de conversa."""
    print("\n" + "="*50)
    print("🤖 GpSRNN-1B Chat Interface")
    print("="*50)
    print("Comandos úteis:")
    print("  /clear  - Limpa o histórico de conversa")
    print("  /quit   - Sai do programa")
    print("  /save   - Salva o estado atual do modelo (debug)")
    print("="*50)
    
    conversation_history = []  # Armazena tokens de toda a conversa
    
    while True:
        try:
            user_input = input("\n👤 Você: ")
            
            if user_input.strip().lower() == '/quit':
                print("👋 Encerrando sessão...")
                break
            
            if user_input.strip().lower() == '/clear':
                conversation_history = []
                print("🧹 Histórico limpo.")
                continue
            
            if user_input.strip().lower() == '/save':
                torch.save(model.state_dict(), 'gp_srrn_debug.pt')
                print("💾 Modelo salvo como 'gp_srrn_debug.pt'")
                continue

            # Tokeniza a entrada do usuário
            new_tokens = tokenizer.encode(user_input, add_special_tokens=False)
            
            # Adiciona ao histórico
            conversation_history.extend(new_tokens)
            
            # Trunca o histórico se for muito longo (janela deslizante simples)
            if len(conversation_history) > max_context_length:
                # Mantém apenas os últimos max_context_length tokens
                conversation_history = conversation_history[-max_context_length:]
                print(f"⚠️  Contexto truncado para {max_context_length} tokens.")

            # Prepara o tensor de entrada
            input_ids = torch.tensor([conversation_history], dtype=torch.long, device=device)
            
            print("🤖 GpSRNN: ", end="", flush=True)
            
            # Geração autoregressiva
            generated_tokens = []
            
            # Estado recorrente inicial (None)
            # O modelo gerencia internamente o estado passo-a-passo quando input_ids tem tamanho 1
            current_ids = input_ids
            
            with torch.autocast(device_type=device if device != 'cpu' else 'cpu', dtype=torch.bfloat16 if device != 'cpu' else torch.float32):
                for _ in range(max_new_tokens):
                    # Se temos histórico longo, passamos tudo na primeira iteração para atualizar o estado interno (se houvesse cache explícito)
                    # Mas nossa implementação de inferência no modelo lida com sequências ou passos únicos.
                    # Para otimização extrema em SSMs, idealmente passaríamos token a token desde o início da conversa,
                    # mas aqui faremos um híbrido: reprocessa o contexto atual para simplificar o código deste script.
                    # NOTA: Em produção, você manteria o estado 'h' persistente entre as mensagens do usuário.
                    
                    logits, _ = model(current_ids) # _ é o estado (não usado explicitamente aqui pois o model lida internamente ou retorna None se batch > 1)
                    
                    # Pega o último logit
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-P Sampling (Nucleus Sampling)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens com probabilidade cumulativa > top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')

                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Verifica token de fim de sequência (EOS)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Imprime o token gerado em tempo real
                    text_gen = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    print(text_gen, end="", flush=True)
                    
                    # Prepara próxima entrada (apenas o último token gerado para eficiência)
                    current_ids = next_token
            
            print() # Nova linha após a resposta
            
            # Adiciona a resposta ao histórico da conversa
            conversation_history.extend(generated_tokens)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupção pelo usuário.")
            break
        except Exception as e:
            print(f"\n❌ Erro durante a geração: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Chat com GpSRNN-1B")
    parser.add_argument('--checkpoint', type=str, default=None, help='Caminho para o arquivo .pt dos pesos treinados')
    parser.add_argument('--max-context', type=int, default=1024, help='Tamanho máximo da janela de contexto')
    parser.add_argument('--max-tokens', type=int, default=256, help='Máximo de tokens gerados por resposta')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperatura para sampling')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("⚠️  GPU não detectada. Rodando em CPU (pode ser lento para 1B parâmetros).")
        print("💡 Dica: No Google Colab, vá em Runtime > Change Runtime Type > GPU T4.")

    model, tokenizer, device = load_model(args.checkpoint)
    
    chat_loop(
        model, 
        tokenizer, 
        device, 
        max_context_length=args.max_context,
        max_new_tokens=args.max_tokens,
        temperature=args.temp
    )

if __name__ == "__main__":
    main()
