"""
Script de treino para o modelo GpSRNN-8M
Treina em dados de texto em português/inglês
"""

import torch
import torch.nn.functional as F
from gp_srrn_8m import (
    GpSRNNConfig, 
    GpSRNNModel, 
    SimpleBPETokenizer,
    create_default_tokenizer,
    get_device_and_dtype,
    count_parameters
)


def prepare_training_data(texts, tokenizer, seq_len=128):
    """Prepara dados de treino a partir de textos."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # Cria pares input/target
    inputs = []
    targets = []
    
    for i in range(0, len(all_tokens) - seq_len, seq_len // 2):
        chunk = all_tokens[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
    
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train_model(
    model,
    tokenizer,
    texts,
    epochs=10,
    batch_size=16,
    seq_len=128,
    lr=3e-4,
    grad_clip=1.0,
    save_path="gp_srrn_8m_checkpoint.pt"
):
    """Treina o modelo GpSRNN."""
    
    device, dtype = get_device_and_dtype()
    model = model.to(device=device, dtype=dtype)
    
    print("=" * 70)
    print("TREINAMENTO GpSRNN-8M")
    print("=" * 70)
    print(f"Dispositivo: {device}")
    print(f"Dtype: {dtype}")
    print(f"Textos: {len(texts)}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Seq len: {seq_len}")
    print(f"Learning rate: {lr}")
    print("=" * 70)
    
    # Prepara dados
    print("\nPreparando dados...")
    input_ids, targets = prepare_training_data(texts, tokenizer, seq_len)
    print(f"Samples de treino: {len(input_ids)}")
    
    # DataLoader manual
    n_samples = len(input_ids)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loop de treino
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Shuffle
        perm = torch.randperm(n_samples)
        input_ids = input_ids[perm]
        targets = targets[perm]
        
        for i in range(0, n_samples, batch_size):
            batch_inputs = input_ids[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            # Forward
            with torch.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', dtype=dtype if dtype != torch.float32 else None):
                logits, _ = model(batch_inputs)
                
                # Loss
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(-1, V),
                    batch_targets.view(-1),
                    ignore_index=-100
                )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Step
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Progresso
            if n_batches % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"\rÉpoca {epoch+1}/{epochs} - Batch {n_batches} - Loss: {avg_loss:.4f}", end="", flush=True)
        
        # Learning rate step
        scheduler.step()
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"\rÉpoca {epoch+1}/{epochs} completa - Loss média: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Salva checkpoint
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model.config,
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint salvo em {save_path}")
    
    print("\n" + "=" * 70)
    print("TREINAMENTO COMPLETO!")
    print("=" * 70)
    
    return model


def generate_sample(model, tokenizer, prompt, max_tokens=100, temperature=0.8):
    """Gera amostra de texto."""
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9
        )
    
    output = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    return output


if __name__ == "__main__":
    # Dados de exemplo para treino (substitua por seus dados reais)
    training_texts = [
        "Olá! Como você está? Eu estou bem, obrigado. E você?",
        "Meu nome é Assistente. Eu sou um modelo de linguagem.",
        "O céu é azul. A grama é verde. O sol é amarelo.",
        "Eu gosto de aprender coisas novas todos os dias.",
        "A vida é bela quando aproveitamos cada momento.",
        "Python é uma linguagem de programação muito popular.",
        "Machine learning é um campo fascinante da ciência da computação.",
        "Inteligência artificial está transformando o mundo.",
        "Bom dia! Espero que você tenha um ótimo dia hoje.",
        "Obrigado pela sua ajuda. Foi muito útil!",
        "Vamos conversar sobre tecnologia e inovação.",
        "Eu adoro ler livros e aprender sobre história.",
        "A natureza é maravilhosa e cheia de surpresas.",
        "Música é uma forma de arte que toca o coração.",
        "Esporte é importante para manter a saúde física e mental.",
        "Comida deliciosa nos faz felizes e satisfeitos.",
        "Viagem é uma experiência enriquecedora e divertida.",
        "Amizade é um tesouro precioso que devemos valorizar.",
        "Família é o pilar fundamental da nossa vida.",
        "Sonhos são importantes para dar sentido à existência.",
        "Hello! How are you today? I hope you're doing well.",
        "My name is Assistant. I am a language model.",
        "The sky is blue. The grass is green. The sun is yellow.",
        "I like to learn new things every single day.",
        "Life is beautiful when we enjoy every moment.",
        "Python is a very popular programming language.",
        "Machine learning is a fascinating field of computer science.",
        "Artificial intelligence is transforming the world.",
        "Good morning! I hope you have a great day today.",
        "Thank you for your help. It was very useful!",
        "Let's talk about technology and innovation.",
        "I love to read books and learn about history.",
        "Nature is wonderful and full of surprises.",
        "Music is an art form that touches the heart.",
        "Sports are important for maintaining physical and mental health.",
        "Delicious food makes us happy and satisfied.",
        "Travel is an enriching and fun experience.",
        "Friendship is a precious treasure we should value.",
        "Family is the fundamental pillar of our lives.",
        "Dreams are important for giving meaning to existence.",
    ] * 100  # Multiplica para ter mais dados
    
    # Duplica mais para ter volume suficiente
    training_texts = training_texts + [
        "O gato pulou no sofá e dormiu tranquilamente.",
        "A criança brincava no parque com seus amigos.",
        "O professor explicou a lição de forma clara.",
        "A loja estava fechada naquele domingo.",
        "O carro novo é rápido e confortável.",
        "A casa tinha um jardim bonito e florido.",
        "O filme foi emocionante do início ao fim.",
        "A receita do bolo ficou perfeita.",
        "O time ganhou o campeonato no último jogo.",
        "A praia estava lotada no verão.",
        "The cat jumped on the sofa and slept peacefully.",
        "The child played in the park with friends.",
        "The teacher explained the lesson clearly.",
        "The store was closed on that Sunday.",
        "The new car is fast and comfortable.",
        "The house had a beautiful flower garden.",
        "The movie was exciting from start to finish.",
        "The cake recipe turned out perfect.",
        "The team won the championship in the last game.",
        "The beach was crowded in the summer.",
    ] * 50
    
    print("Criando configuração e modelo...")
    config = GpSRNNConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=8,
        n_heads=4,
        d_ffn=512,
        dropout=0.1,
        max_seq_len=256
    )
    
    model = GpSRNNModel(config)
    
    params = count_parameters(model)
    print(f"Parâmetros do modelo: {params['total_millions']:.2f}M")
    
    print("\nCriando tokenizer...")
    tokenizer = create_default_tokenizer()
    print(f"Vocabulário: {len(tokenizer.token_to_id)} tokens")
    
    print("\nIniciando treino...\n")
    trained_model = train_model(
        model=model,
        tokenizer=tokenizer,
        texts=training_texts,
        epochs=20,
        batch_size=8,
        seq_len=64,
        lr=5e-4,
        save_path="gp_srrn_8m_trained.pt"
    )
    
    # Testa geração após treino
    print("\n" + "=" * 70)
    print("TESTANDO GERAÇÃO APÓS TREINO")
    print("=" * 70)
    
    prompts = [
        "Olá! Como",
        "O céu é",
        "Eu gosto de",
        "Hello! How",
        "The sun is",
        "I like to",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        output = generate_sample(trained_model, tokenizer, prompt, max_tokens=30, temperature=0.7)
        print(f"Geração: {output}")
    
    print("\n" + "=" * 70)
    print("Modelo treinado salvo em: gp_srrn_8m_trained.pt")
    print("Para usar no chat: python gp_srrn_8m.py --chat gp_srrn_8m_trained.pt")
    print("=" * 70)
