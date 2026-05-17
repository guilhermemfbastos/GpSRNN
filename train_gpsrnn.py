"""
Script de Treinamento para GpSRNN-8M
Otimizado para Google Colab (T4 Free Tier)

Uso:
    python train_gpsrnn.py --data corpus.txt --epochs 10 --batch-size 32

Ou no Colab:
    !python train_gpsrnn.py --data corpus.txt --epochs 5 --batch-size 64
"""

import argparse
import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gp_srrn_8m import GpSRNNConfig, GpSRNNModel, SimpleBPETokenizer


class TextDataset(Dataset):
    """Dataset de texto para treinamento"""
    
    def __init__(self, text: str, tokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenizar todo o texto SEM tokens especiais (bos/eos)
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Calcular número de sequências
        self.n_sequences = max(0, len(self.tokens) - seq_length - 1)
        
        print(f"  Tokens totais: {len(self.tokens):,}")
        print(f"  Sequências de treino: {self.n_sequences:,}")
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Extrair sequência
        start = idx
        end = idx + self.seq_length + 1  # +1 para o target
        
        chunk = self.tokens[start:end]
        
        # Input: tokens[0:seq_length]
        # Target: tokens[1:seq_length+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def load_training_data(data_path: str, min_chars: int = 10000) -> str:
    """Carrega dados de treinamento de arquivo(s)"""
    
    if os.path.isdir(data_path):
        # Carregar todos os arquivos .txt do diretório
        files = list(Path(data_path).glob("*.txt"))
        if not files:
            raise ValueError(f"Nenhum arquivo .txt encontrado em {data_path}")
        
        texts = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                texts.append(file.read())
        
        text = "\n".join(texts)
        print(f"Carregados {len(files)} arquivos")
    
    elif os.path.isfile(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    else:
        # Dados de exemplo (português + inglês)
        print("Usando dados de exemplo...")
        text = """
        Olá! Como você está? Eu sou um modelo de linguagem chamado GpSRNN.
        Estou aprendendo a gerar texto em português e inglês.
        The quick brown fox jumps over the lazy dog.
        Machine learning is a subset of artificial intelligence.
        O Brasil é um país maravilhoso com muita diversidade cultural.
        Deep learning models can learn from large amounts of data.
        A inteligência artificial está transformando o mundo.
        Natural language processing enables computers to understand human language.
        Vamos treinar este modelo por várias épocas para melhorar sua performance.
        Training neural networks requires patience and computational resources.
        """ * 100  # Repetir para ter dados suficientes
    
    if len(text) < min_chars:
        print(f"Aviso: Texto muito curto ({len(text)} chars). Repetindo para aumentar...")
        multiplier = (min_chars // len(text)) + 1
        text = text * multiplier
    
    return text


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip: float = 1.0):
    """Treina uma época completa"""
    
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # Mover para dispositivo
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        logits, _ = model(x)  # [batch, seq_len, vocab_size]
        
        # Calcular loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Log progresso
        if (batch_idx + 1) % 100 == 0 or batch_idx == len(dataloader) - 1:
            avg_loss = total_loss / n_batches
            elapsed = time.time() - start_time
            tokens_per_sec = (n_batches * dataloader.batch_size * x.size(1)) / elapsed
            
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 10):
    """Avalia o modelo no dataset de validação"""
    
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        logits, _ = model(x)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt: str = "Olá", max_tokens: int = 50, 
                   temperature: float = 0.8, top_k: int = 40):
    """Gera uma amostra de texto para monitorar progresso"""
    
    model.eval()
    
    # Tokenizar prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.token_to_id.get('<bos>', 0)]
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Gerar
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )[0]
    
    # Decodificar
    generated_text = tokenizer.decode(output_ids.tolist())
    
    return generated_text


def save_checkpoint(model, tokenizer, optimizer, epoch, loss, path: str):
    """Salva checkpoint do treinamento"""
    
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'vocab': tokenizer.token_to_id,
        'merges': tokenizer.merges,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint salvo em {path}")


def load_checkpoint(path: str, device):
    """Carrega checkpoint do treinamento"""
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = GpSRNNModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = SimpleBPETokenizer()
    tokenizer.token_to_id = checkpoint['vocab']
    tokenizer.id_to_token = {v: k for k, v in checkpoint['vocab'].items()}
    tokenizer.merges = checkpoint['merges']
    
    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Treinar GpSRNN-8M")
    
    # Dados
    parser.add_argument("--data", type=str, default="",
                       help="Caminho para arquivo .txt ou diretório com dados")
    parser.add_argument("--seq-length", type=int, default=128,
                       help="Comprimento da sequência de treino")
    parser.add_argument("--val-split", type=float, default=0.05,
                       help="Fração dos dados para validação")
    
    # Hiperparâmetros
    parser.add_argument("--epochs", type=int, default=10,
                       help="Número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps para LR scheduler")
    
    # Modelo
    parser.add_argument("--vocab-size", type=int, default=8192,
                       help="Tamanho do vocabulário do tokenizer")
    parser.add_argument("--d-model", type=int, default=256,
                       help="Dimensão do modelo")
    parser.add_argument("--n-layers", type=int, default=8,
                       help="Número de camadas")
    parser.add_argument("--n-heads", type=int, default=8,
                       help="Número de heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                       help="Diretório para salvar checkpoints")
    parser.add_argument("--save-every", type=int, default=1,
                       help="Salvar checkpoint a cada N épocas")
    parser.add_argument("--sample-every", type=int, default=1,
                       help="Gerar amostra de texto a cada N épocas")
    parser.add_argument("--resume", type=str, default="",
                       help="Caminho para checkpoint para resumir treino")
    
    # Dispositivo
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Dispositivo para treino")
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            print(f"🚀 Usando CUDA ({dtype})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
            print(f"🍎 Usando MPS (Mac)")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            print(f"💻 Usando CPU")
    else:
        device = torch.device(args.device)
        dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    # Criar diretório de output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Carregar dados
    print("\n📚 Carregando dados de treinamento...")
    text = load_training_data(args.data)
    print(f"  Caracteres totais: {len(text):,}")
    
    # Dividir train/val
    val_size = int(len(text) * args.val_split)
    train_text = text[val_size:]
    val_text = text[:val_size]
    
    print(f"  Treino: {len(train_text):,} chars")
    print(f"  Validação: {len(val_text):,} chars")
    
    # Criar tokenizer
    print("\n🔤 Criando tokenizer...")
    tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)
    # Divide o texto em chunks para treinamento
    chunk_size = 1000
    texts = [text[i:i+chunk_size] for i in range(0, min(len(text), 50000), chunk_size)]
    tokenizer.train(texts, target_vocab_size=args.vocab_size)
    print(f"  Vocabulário: {len(tokenizer.token_to_id)} tokens")
    
    # Criar datasets
    print("\n📊 Criando datasets...")
    train_dataset = TextDataset(train_text, tokenizer, args.seq_length)
    val_dataset = TextDataset(val_text, tokenizer, args.seq_length)
    
    # Verificar se há dados suficientes
    if len(train_dataset) == 0:
        print("❌ Erro: Dados de treino insuficientes!")
        print(f"   Texto de treino tem {len(train_text)} caracteres")
        print(f"   Mas apenas {len(tokenizer.encode(train_text))} tokens foram gerados")
        print(f"   Tente reduzir seq_length (atual: {args.seq_length}) ou aumentar os dados")
        return
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Criar modelo
    print("\n🏗️ Criando modelo...")
    config = GpSRNNConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    )
    
    model = GpSRNNModel(config).to(dtype=dtype)
    model = model.to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parâmetros totais: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Parâmetros treináveis: {trainable_params:,}")
    
    # Resume de checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\n📥 Carregando checkpoint: {args.resume}")
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        model = model.to(dtype=dtype)
        model = model.to(device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumindo da época {start_epoch}")
    
    # Otimizador e scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler com warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Se resuming, carregar estados do optimizer/scheduler
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Loop de treinamento
    print("\n" + "="*70)
    print("🚀 INICIANDO TREINAMENTO")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"ÉPOCA {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
        epoch_start = time.time()
        
        # Treinar
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, grad_clip=args.grad_clip
        )
        
        # Avaliar
        val_loss = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Resultados da Época {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Tempo:      {epoch_time:.1f}s")
        print(f"  LR atual:   {scheduler.get_last_lr()[0]:.6f}")
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, "best_model.pt")
            )
            print(f"  ✨ Novo melhor modelo! Val loss: {val_loss:.4f}")
        
        # Salvar checkpoint periódico
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, tokenizer, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )
        
        # Gerar amostra de texto
        if (epoch + 1) % args.sample_every == 0:
            print(f"\n📝 Amostra de texto:")
            prompts = ["Olá", "The", "O Brasil", "Machine"]
            for prompt in prompts:
                try:
                    sample = generate_sample(
                        model, tokenizer, device,
                        prompt=prompt,
                        max_tokens=40,
                        temperature=0.8
                    )
                    print(f"  '{prompt}' → {sample[:100]}...")
                except Exception as e:
                    print(f"  Erro ao gerar: {e}")
    
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"Melhor val loss: {best_val_loss:.4f}")
    print(f"Checkpoints salvos em: {args.output_dir}/")
    print("="*70)
    
    # Teste final
    print("\n🧪 Teste final de geração:")
    sample = generate_sample(
        model, tokenizer, device,
        prompt="Olá, como você está?",
        max_tokens=60,
        temperature=0.7
    )
    print(sample)


if __name__ == "__main__":
    main()
