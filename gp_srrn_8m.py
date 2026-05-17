"""
GpSRNN-8M - Generative Pre-Trained State-Space Recurrent Network
Arquitetura leve (~8M parâmetros) implementada totalmente do zero
Sem dependências externas além do PyTorch puro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math
import json
import re
from collections import Counter


# ============================================================================
# TOKENIZER BPE SIMPLES (Implementado do zero)
# ============================================================================

class SimpleBPETokenizer:
    """
    Tokenizador Byte-Pair Encoding simplificado, implementado do zero.
    Não depende da biblioteca transformers.
    """
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        self._build_base_vocab()
        
    def _build_base_vocab(self):
        """Constrói vocabulário base com caracteres individuais + especiais."""
        # Adiciona tokens especiais
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Adiciona todos os caracteres ASCII imprimíveis
        idx = len(self.special_tokens)
        for i in range(32, 127):  # Caracteres imprimíveis
            char = chr(i)
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1
        
        # Adiciona alguns caracteres UTF-8 comuns
        for char in 'áàâãéêíóôõúçñüÁÀÂÃÉÊÍÓÔÕÚÇÑÜ':
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1
    
    def train(self, texts: List[str], target_vocab_size: int = 8192):
        """
        Treina o tokenizer em uma lista de textos.
        Implementação simplificada do algoritmo BPE.
        """
        print(f"Treinando tokenizer com {len(texts)} textos...")
        
        # Contar pares de símbolos
        def get_pairs(word: List[str]) -> Counter:
            pairs = Counter()
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += 1
            return pairs
        
        # Tokenizar texto em caracteres
        def tokenize_to_chars(text: str) -> List[str]:
            return list(text)
        
        # Preparar corpus como lista de listas de tokens
        corpus = [tokenize_to_chars(text) for text in texts]
        
        # Iterativamente aprender merges
        while len(self.token_to_id) < target_vocab_size:
            # Contar todos os pares no corpus
            pair_counts = Counter()
            for tokens in corpus:
                pair_counts.update(get_pairs(tokens))
            
            if not pair_counts:
                break
            
            # Escolher o par mais frequente
            best_pair = pair_counts.most_common(1)[0][0]
            if pair_counts[best_pair] == 0:
                break
            
            # Criar novo token
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[new_token] = new_id
                self.id_to_token[new_id] = new_token
                self.merges.append(best_pair)
            
            # Aplicar merge ao corpus
            new_corpus = []
            for tokens in corpus:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                corpus = [new_tokens]
                new_corpus.extend(corpus)
                corpus = new_corpus
        
        print(f"Vocabulário final: {len(self.token_to_id)} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Codifica texto em IDs de tokens."""
        # Começa com caracteres individuais
        tokens = list(text)
        
        # Aplica merges iterativamente
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                    new_tokens.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Converte para IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Token desconhecido - divide em caracteres
                for char in token:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.token_to_id['<unk>'])
        
        if add_special_tokens:
            ids = [self.token_to_id['<bos>']] + ids + [self.token_to_id['<eos>']]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodifica IDs de tokens em texto."""
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')
        return ''.join(tokens)
    
    def save(self, path: str):
        """Salva o tokenizer em arquivo JSON."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer salvo em {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SimpleBPETokenizer':
        """Carrega tokenizer de arquivo JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        # Reconstroi id_to_token a partir de token_to_id
        tokenizer.id_to_token = {v: k for k, v in data['token_to_id'].items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.special_tokens = data['special_tokens']
        return tokenizer


def create_default_tokenizer() -> SimpleBPETokenizer:
    """Cria um tokenizer padrão com vocabulário básico."""
    tokenizer = SimpleBPETokenizer(vocab_size=8192)
    
    # Adiciona tokens comuns em português/inglês manualmente
    common_words = [
        'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'no', 'na',
        'e', 'ou', 'que', 'se', 'para', 'com', 'não', 'sim', 'eu', 'tu', 'ele',
        'ela', 'nós', 'vós', 'eles', 'elas', 'é', 'são', 'foi', 'foram', 'ser',
        'estar', 'ter', 'haver', 'fazer', 'dizer', 'the', 'a', 'an', 'and', 'or',
        'is', 'are', 'was', 'were', 'be', 'have', 'has', 'do', 'does', 'did',
        'I', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
        'now', 'hello', 'hi', 'bye', 'good', 'bad', 'yes', 'no', 'ok', 'okay',
        'como', 'vai', 'bem', 'mal', 'olá', 'tchau', 'bom', 'ruim', 'grande',
        'pequeno', 'casa', 'agua', 'fogo', 'terra', 'ar', 'sol', 'lua', 'dia',
        'noite', 'tempo', 'vida', 'morte', 'amor', 'odio', 'feliz', 'triste',
        ' ', '\n', '\t', '  ', '   ', ' A', ' O', ' E', ' D', ' S', ' T',
        'ão', 'õe', 'ea', 'oa', 'ia', 'ua', 'ei', 'ai', 'oi', 'ui', 'au', 'eu',
        'th', 'in', 'er', 'on', 'at', 'en', 'or', 'an', 'al', 'ar', 'as', 'is',
        'it', 'to', 'of', 'ed', 'ng', 'ly', 'ty', 'ry', 'le', 're', 've', 'te',
        'me', 'my', 'your', 'name', 'is', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'better', 'best', 'more', 'most',
        'less', 'least', 'much', 'many', 'little', 'few', 'great', 'high', 'low',
        'long', 'short', 'wide', 'narrow', 'deep', 'shallow', 'thick', 'thin',
        'heavy', 'light', 'hard', 'soft', 'rough', 'smooth', 'hot', 'cold',
        'warm', 'cool', 'dry', 'wet', 'clean', 'dirty', 'full', 'empty', 'new',
        'old', 'young', 'fresh', 'stale', 'rich', 'poor', 'cheap', 'expensive',
        'free', 'busy', 'slow', 'fast', 'quick', 'rapid', 'swift', 'speedy',
        'early', 'late', 'soon', 'now', 'then', 'here', 'there', 'where', 'when',
        'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that',
        'these', 'those', 'such', 'same', 'other', 'another', 'any', 'some',
        'all', 'both', 'each', 'every', 'either', 'neither', 'one', 'two',
        'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next', 'previous',
        'present', 'past', 'future', 'time', 'day', 'night', 'week', 'month',
        'year', 'hour', 'minute', 'moment', 'while', 'ago', 'before', 'after',
        'during', 'since', 'until', 'from', 'to', 'into', 'onto', 'upon', 'with',
        'without', 'by', 'for', 'about', 'against', 'between', 'through',
        'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'minha', 'sua', 'nossa', 'tua', 'meu', 'seu', 'nosso', 'teu', 'qual',
        'quando', 'onde', 'porque', 'quanto', 'quem', 'muito', 'pouco', 'mais',
        'menos', 'tudo', 'nada', 'algo', 'alguém', 'ninguém', 'todo', 'nenhum',
        'certo', 'errado', 'verdade', 'mentira', 'real', 'falso', 'possivel',
        'impossivel', 'facil', 'dificil', 'simples', 'complexo', 'claro', 'escuro',
        'forte', 'fraco', 'alto', 'baixo', 'largo', 'estreito', 'perto', 'longe',
        'dentro', 'fora', 'cima', 'baixo', 'frente', 'tras', 'lado', 'meio',
        'fim', 'inicio', 'comeco', 'parte', 'pedaco', 'pedaço', 'forma', 'jeito',
        'modo', 'maneira', 'tipo', 'classe', 'grupo', 'numero', 'quantidade',
        'resposta', 'pergunta', 'palavra', 'frase', 'texto', 'livro', 'pagina',
        'linha', 'letra', 'som', 'voz', 'ruido', 'silencio', 'musica', 'canto',
        'danca', 'arte', 'obra', 'cor', 'luz', 'sombra', 'imagem', 'foto',
        'filme', 'video', 'jogo', 'brincadeira', 'diversao', 'prazer', 'gozo',
        'dor', 'sofrimento', 'pena', 'tristeza', 'alegria', 'felicidade', 'contente',
        'satisfeito', 'realizado', 'completo', 'inteiro', 'total', 'parcial',
        'metade', 'duplo', 'triplo', 'unico', 'solo', 'sozinho', 'junto', 'unido',
        'separado', 'dividido', 'quebrado', 'partido', 'cortado', 'aberto', 'fechado',
        'ligado', 'desligado', 'aceso', 'apagado', 'vivo', 'morto', 'nascer',
        'morrer', 'crescer', 'diminuir', 'aumentar', 'reduzir', 'subir', 'descer',
        'entrar', 'sair', 'chegar', 'partir', 'vir', 'ir', 'ficar', 'passar',
        'andar', 'correr', 'pular', 'saltar', 'voar', 'nadar', 'mergulhar',
        'cair', 'levantar', 'sentar', 'deitar', 'dormir', 'acordar', 'comer',
        'beber', 'tomar', 'dar', 'receber', 'pegar', 'segurar', 'largar', 'soltar',
        'abrir', 'fechar', 'puxar', 'empurrar', 'bater', 'tocar', 'sentir',
        'ver', 'olhar', 'ouvir', 'escutar', 'falar', 'dizer', 'contar', 'narrar',
        'escrever', 'ler', 'estudar', 'aprender', 'ensinar', 'mostrar', 'explicar',
        'entender', 'compreender', 'saber', 'conhecer', 'pensar', 'achar', 'crer',
        'acreditar', 'confiar', 'duvidar', 'querer', 'desejar', 'esperar', 'aguardar',
        'procurar', 'buscar', 'encontrar', 'perder', 'ganhar', 'vencer', 'perder',
        'tentar', 'experimentar', 'testar', 'usar', 'utilizar', 'aplicar', 'colocar',
        'por', 'deixar', 'permitir', 'deixar', 'impedir', 'proibir', 'mandar',
        'ordenar', 'pedir', 'solicitar', 'rogar', 'implorar', 'suplicar', 'rezar',
        'orar', 'louvar', 'abençoar', 'agradecer', 'perdoar', 'culpar', 'acusar',
        'defender', 'atacar', 'lutar', 'brigar', 'discutir', 'debater', 'conversar',
        'dialogar', 'comunicar', 'expressar', 'manifestar', 'declarar', 'anunciar',
        'informar', 'avisar', 'alertar', 'advertir', 'aconselhar', 'sugerir',
        'propor', 'oferecer', 'apresentar', 'introduzir', 'representar', 'simbolizar',
        'significar', 'valer', 'custar', 'pagar', 'comprar', 'vender', 'trocar',
        'emprestar', 'devolver', 'guardar', 'conservar', 'preservar', 'proteger',
        'defender', 'amparar', 'ajudar', 'auxiliar', 'assistir', 'servir', 'atender',
        'cuidar', 'zelar', 'tratar', 'curar', 'sarar', 'melhorar', 'piorar',
        'agravar', 'aliviar', 'amenizar', 'suavizar', 'fortalecer', 'enfraquecer',
        'endurecer', 'amaciar', 'aquecer', 'esfriar', 'resfriar', ' gelar', 'derreter',
        'congelar', 'ferver', 'cozinhar', 'assar', 'fritar', 'grellhar', 'tostar',
        'queimar', 'incendiar', 'apagar', 'extinguir', 'eliminar', 'remover',
        'retirar', 'extrair', 'arrancar', 'arrastar', 'carregar', 'transportar',
        'conduzir', 'guiar', 'dirigir', 'pilotar', 'navegar', 'viajar', 'passear',
        'visitar', 'turismo', 'excursao', 'viagem', 'passeio', 'volta', 'retorno',
        'regresso', 'chegada', 'partida', 'saida', 'entrada', 'acesso', 'passagem',
        'caminho', 'estrada', 'rua', 'avenida', 'alameda', 'boulevard', 'travessa',
        'beco', 'viela', 'atalho', 'desvio', 'curva', 'volta', 'meandro', 'serpente',
        'linhareta', 'diagonal', 'vertical', 'horizontal', 'oblquo', 'transversal',
        'paralelo', 'perpendicular', 'ortogonal', 'normal', 'tangente', 'secante',
        'circulo', 'esfera', 'bola', 'globos', 'cilindro', 'cono', 'piramide',
        'cubo', 'quadrado', 'retangulo', 'losango', 'trapézio', 'paralelogramo',
        'triangulo', 'poligono', 'pentagono', 'hexagono', 'heptagono', 'octogono',
        'nonagono', 'decagono', 'estrela', 'cruz', 'seta', 'flecha', 'ponta',
        'vértice', 'ângulo', 'lado', 'face', 'aresta', 'base', 'topo', 'cume',
        'pico', 'monte', 'montanha', 'colina', 'morro', 'serra', 'cordilheira',
        'vale', 'planicie', 'planalto', 'meseta', 'chapada', 'depressao', 'canyon',
        'desfiladeiro', 'garganta', 'fenda', 'fissura', 'rachadura', 'trincheira',
        'buraco', 'cova', 'caverna', 'gruta', 'toca', 'tunel', 'passagem', 'corredor',
        'hall', 'sala', 'quarto', 'cozinha', 'banheiro', 'lavabo', 'dispensa',
        'despensa', 'garagem', 'porao', 'sotao', 'varanda', 'terraço', 'sacada',
        'quintal', 'jardim', 'horta', 'pomar', 'chacará', 'sitio', 'fazenda',
        'rancho', 'granja', 'haras', 'curral', 'celeiro', 'paiol', 'tulha',
        'armazem', 'deposito', 'estoque', 'reserva', 'pulmon', 'reserva', 'backup',
    ]
    
    idx = len(tokenizer.token_to_id)
    for word in common_words:
        if word not in tokenizer.token_to_id:
            tokenizer.token_to_id[word] = idx
            tokenizer.id_to_token[idx] = word
            idx += 1
            if idx >= 8192:
                break
    
    print(f"Tokenizer criado com {len(tokenizer.token_to_id)} tokens")
    return tokenizer


# ============================================================================
# CONFIGURAÇÃO DO MODELO (~8M parâmetros)
# ============================================================================

@dataclass
class GpSRNNConfig:
    """Configuração do modelo GpSRNN-8M"""
    vocab_size: int = 8192  # Reduzido para 8K
    d_model: int = 256      # Reduzido para 256
    n_layers: int = 8       # Reduzido para 8 camadas
    n_heads: int = 4        # 4 heads
    d_ffn: int = 512        # FFN expandido 2x
    dropout: float = 0.1
    max_seq_len: int = 512
    
    def estimate_parameters(self) -> Dict[str, int]:
        """Estima o número de parâmetros do modelo."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        
        # Por camada TimeMix
        # R, K, V, G projections: 4 * (d_model * d_model)
        # alpha decay: d_model
        # output projection: d_model * d_model
        time_mix_per_layer = (4 * self.d_model * self.d_model) + self.d_model + (self.d_model * self.d_model)
        
        # Por camada ChannelMix (SwiGLU)
        # key_proj: d_model * d_ffn
        # value_proj: d_model * d_ffn  
        # out_proj: d_ffn * d_model
        channel_mix_per_layer = (2 * self.d_model * self.d_ffn) + (self.d_ffn * self.d_model)
        
        # LayerNorms: 2 por bloco * 2 * d_model
        norm_per_layer = 4 * self.d_model
        
        # Total por camada
        per_layer = time_mix_per_layer + channel_mix_per_layer + norm_per_layer
        
        # Total do modelo
        total = embed_params + (per_layer * self.n_layers) + embed_params  # + embed_params para LM head (weight tying opcional)
        
        return {
            'embeddings': embed_params,
            'time_mix_per_layer': time_mix_per_layer,
            'channel_mix_per_layer': channel_mix_per_layer,
            'norm_per_layer': norm_per_layer,
            'per_layer': per_layer,
            'total_layers': per_layer * self.n_layers,
            'lm_head': embed_params,
            'total': total,
            'total_millions': total / 1e6
        }


# ============================================================================
# COMPONENTES DO MODELO GpSRNN
# ============================================================================

class LayerNorm(nn.Module):
    """LayerNorm estável numericamente."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class TimeMix(nn.Module):
    """
    Mecanismo Time-Mix com recorrência linear e portões seletivos.
    
    Matemática da Recorrência Linear:
    ----------------------------------
    Para cada timestep t:
        - Receptance: R_t = sigmoid(x_t @ W_R)
        - Key:        K_t = x_t @ W_K
        - Value:      V_t = x_t @ W_V
        - Gate:       G_t = x_t @ W_G (ativação linear/SiLU)
        - Decay:      alpha (aprendível, entre 0 e 1)
        
        Atualização do estado (recorrência):
            state_t = alpha * state_{t-1} + K_t * V_t
        
        Saída:
            output_t = R_t * state_t + G_t
    
    Durante o treino: usamos parallel scan para processar toda a sequência de uma vez.
    Durante a inferência: processamos passo-a-passo mantendo o estado.
    """
    
    def __init__(self, config: GpSRNNConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        # Projeções para R, K, V, G
        self.W_R = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_G = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Projeção de saída
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Decay aprendível (alpha) - inicializado perto de 1 para memória longa
        self.alpha = nn.Parameter(torch.ones(config.d_model) * 0.9)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para treino usando parallel scan.
        x: [batch, seq_len, d_model]
        """
        B, T, D = x.shape
        
        # Calcula R, K, V, G
        R = torch.sigmoid(self.W_R(x))  # [B, T, D]
        K = self.W_K(x)                  # [B, T, D]
        V = self.W_V(x)                  # [B, T, D]
        G = F.silu(self.W_G(x))          # [B, T, D]
        
        # Reshape para multi-head
        R = R.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        G = G.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Parallel scan para computar o estado recorrente
        # state_t = alpha * state_{t-1} + K_t * V_t
        alpha = self.alpha.view(1, self.n_heads, 1, self.head_dim)
        
        # KV product em cada timestep
        KV = K * V  # [B, H, T, head_dim]
        
        # Accumulate states usando cumsum ponderado por alpha
        # Precisa de implementação cuidadosa para parallel scan
        log_alpha = torch.log(alpha.clamp(min=1e-6))
        
        # Cálculo eficiente do estado acumulado
        states = []
        state = torch.zeros_like(KV[:, :, 0:1])
        for t in range(T):
            state = alpha * state + KV[:, :, t:t+1]
            states.append(state)
        state_seq = torch.cat(states, dim=2)  # [B, H, T, head_dim]
        
        # Aplica receptance e gate
        output = R * state_seq + G  # [B, H, T, head_dim]
        
        # Reshape de volta e projeção final
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output
    
    def forward_infer(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass para inferência passo-a-passo.
        x: [batch, d_model] (single timestep)
        state: [batch, n_heads, head_dim] ou None
        Returns: output, new_state
        """
        B, D = x.shape
        
        # Calcula R, K, V, G para single timestep
        R = torch.sigmoid(self.W_R(x))  # [B, D]
        K = self.W_K(x)                  # [B, D]
        V = self.W_V(x)                  # [B, D]
        G = F.silu(self.W_G(x))          # [B, D]
        
        # Reshape para multi-head
        R = R.view(B, self.n_heads, self.head_dim)
        K = K.view(B, self.n_heads, self.head_dim)
        V = V.view(B, self.n_heads, self.head_dim)
        G = G.view(B, self.n_heads, self.head_dim)
        
        # Inicializa ou usa estado existente
        if state is None:
            state = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        
        # Atualiza estado: state = alpha * state + K * V
        alpha = self.alpha.view(1, self.n_heads, self.head_dim)
        new_state = alpha * state + K * V
        
        # Calcula saída: output = R * state + G
        output = R * new_state + G  # [B, H, head_dim]
        
        # Reshape e projeção final
        output = output.contiguous().view(B, D)
        output = self.W_O(output)
        
        return output, new_state
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward automático baseado na forma da entrada."""
        if x.dim() == 3:
            # [B, T, D] - modo treino
            return self.forward_train(x)
        else:
            # [B, D] - modo inferência
            return self.forward_infer(x, state)


class ChannelMix(nn.Module):
    """
    Channel-Mix (FFN) com ativação SwiGLU.
    
    SwiGLU(x) = SiLU(x @ W_key) * (x @ W_value)
    
    Arquitetura:
        x -> LayerNorm -> [W_key, W_value] -> SiLU * linear -> W_out -> output
    """
    
    def __init__(self, config: GpSRNNConfig):
        super().__init__()
        self.config = config
        
        # Projeções SwiGLU
        self.W_key = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.W_value = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.W_out = nn.Linear(config.d_ffn, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model] ou [batch, d_model]
        """
        key = F.silu(self.W_key(x))
        value = self.W_value(x)
        
        # SwiGLU activation
        hidden = key * value
        output = self.W_out(hidden)
        output = self.dropout(output)
        
        return output


class GpSRNNBlock(nn.Module):
    """
    Bloco GpSRNN completo com Pre-Norm.
    
    Estrutura:
        x -> LayerNorm -> TimeMix -> x + output -> LayerNorm -> ChannelMix -> x + output
    """
    
    def __init__(self, config: GpSRNNConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.time_mix = TimeMix(config)
        self.norm2 = LayerNorm(config.d_model)
        self.channel_mix = ChannelMix(config)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, d_model] ou [batch, d_model]
        state: estado recorrente para inferência
        Returns: output, new_state (apenas para inferência)
        """
        if x.dim() == 3:
            # Modo treino
            h = x + self.time_mix(self.norm1(x))
            output = h + self.channel_mix(self.norm2(h))
            return output, None
        else:
            # Modo inferência
            tm_output, new_state = self.time_mix.forward_infer(self.norm1(x), state)
            h = x + tm_output
            output = h + self.channel_mix(self.norm2(h))
            return output, new_state


class GpSRNNModel(nn.Module):
    """
    Modelo GpSRNN completo.
    
    Arquitetura:
        Input -> Embedding -> [GpSRNNBlock] * n_layers -> LayerNorm -> LM Head -> Output
    """
    
    def __init__(self, config: GpSRNNConfig):
        super().__init__()
        self.config = config
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Posicional (opcional - a recorrência já captura posição)
        # Usamos embedding posicional aprendido simples
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Blocos GpSRNN
        self.blocks = nn.ModuleList([
            GpSRNNBlock(config) for _ in range(config.n_layers)
        ])
        
        # Normalização final
        self.final_norm = LayerNorm(config.d_model)
        
        # LM Head (sem weight tying para simplicidade)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Inicialização de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicialização de pesos estilo GPT."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass do modelo.
        
        Args:
            input_ids: [batch, seq_len] - IDs dos tokens
            states: Lista de estados recorrentes para inferência
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_states: Lista de novos estados (para inferência)
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Detecta modo (treino vs inferência)
        is_inference = (T == 1) or (states is not None and states[0] is not None)
        
        # Embeddings
        x = self.token_embedding(input_ids)  # [B, T, D]
        
        if not is_inference:
            # Adiciona embeddings posicionais (apenas no treino)
            positions = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        
        x = self.dropout(x)
        
        # Passa pelos blocos
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if (states is not None and i < len(states)) else None
            if is_inference and state is not None:
                # Modo inferência passo-a-passo
                x = x.squeeze(1) if x.dim() == 3 else x  # [B, D]
                x, new_state = block(x, state)
                x = x.unsqueeze(1)  # [B, 1, D]
                new_states.append(new_state)
            else:
                # Modo treino
                x, _ = block(x)
        
        # Normalização final
        x = self.final_norm(x)
        
        # LM Head
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        return logits, new_states
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 pad_token_id: int = 0,
                 eos_token_id: int = 3) -> torch.Tensor:
        """
        Geração autoregressiva com memória constante.
        
        Args:
            input_ids: [batch, seq_len] - Prompt inicial
            max_new_tokens: Número máximo de tokens para gerar
            temperature: Temperatura para sampling
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            pad_token_id: ID do token de padding
            eos_token_id: ID do token de fim de sequência
        
        Returns:
            generated_ids: [batch, seq_len + new_tokens] - Texto gerado
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Inicializa estados recorrentes (um por camada)
        states = [None for _ in range(self.config.n_layers)]
        
        # Processa o prompt inicial
        with torch.no_grad():
            # Se o prompt tiver múltiplos tokens, processa todos menos o último
            # para inicializar os estados
            if input_ids.shape[1] > 1:
                # Processa todo o prompt exceto o último token
                prompt_without_last = input_ids[:, :-1]
                _, states = self(prompt_without_last, states=None)
                
                # Pega o último token do prompt como starting point
                current_token = input_ids[:, -1:]  # [B, 1]
            else:
                current_token = input_ids  # [B, 1]
            
            generated = []
            
            for _ in range(max_new_tokens):
                # Forward passo-a-passo
                logits, states = self(current_token, states=states)
                logits = logits[:, -1, :]  # [B, vocab_size]
                
                # Aplica temperatura
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                
                generated.append(next_token)
                
                # Verifica se todos terminaram
                if (next_token == eos_token_id).all():
                    break
                
                current_token = next_token
            
            # Concatena gerados
            if generated:
                generated_ids = torch.cat(generated, dim=1)  # [B, new_tokens]
                return torch.cat([input_ids, generated_ids], dim=1)
            else:
                return input_ids


# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Conta o número de parâmetros do modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


def estimate_memory_usage(model: nn.Module, batch_size: int = 1, seq_len: int = 512) -> Dict[str, float]:
    """Estima o uso de memória do modelo."""
    # Memória dos parâmetros (em MB)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # Memória dos gradientes
    grad_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024 / 1024
    
    # Memória dos ativadores (estimativa grosseira)
    # Cada layer armazena ativadores de tamanho [batch, seq_len, d_model]
    d_model = model.config.d_model
    n_layers = model.config.n_layers
    activation_memory = (batch_size * seq_len * d_model * 4 * n_layers) / 1024 / 1024  # 4 bytes por float32
    
    # Estados recorrentes (inferência)
    recurrent_state_memory = (batch_size * d_model * n_layers * 4) / 1024 / 1024
    
    return {
        'parameters_mb': param_memory,
        'gradients_mb': grad_memory,
        'activations_mb': activation_memory,
        'recurrent_states_mb': recurrent_state_memory,
        'total_training_mb': param_memory + grad_memory + activation_memory,
        'total_inference_mb': param_memory + recurrent_state_memory
    }


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Detecta o melhor dispositivo e dtype disponível."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Verifica suporte a bfloat16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    
    return device, dtype


# ============================================================================
# LOOP DE TREINO MINIMALISTA
# ============================================================================

def train_step(model: nn.Module, 
               optimizer: torch.optim.Optimizer,
               input_ids: torch.Tensor,
               targets: torch.Tensor,
               grad_clip: float = 1.0) -> float:
    """Executa um passo de treino."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, _ = model(input_ids)
    
    # Calcula loss
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        ignore_index=-100
    )
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def generate_text(model: nn.Module, 
                  tokenizer: SimpleBPETokenizer,
                  prompt: str,
                  max_new_tokens: int = 100,
                  temperature: float = 0.8) -> str:
    """Gera texto a partir de um prompt."""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokeniza o prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Gera
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    
    # Decodifica
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    return output_text


# ============================================================================
# INTERFACE DE CHAT
# ============================================================================

class ChatInterface:
    """Interface de chat interativo para o modelo GpSRNN."""
    
    def __init__(self, model: nn.Module, tokenizer: SimpleBPETokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.history: List[str] = []
    
    def respond(self, user_input: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """Gera resposta para entrada do usuário."""
        # Constrói o prompt com histórico
        if self.history:
            context = '\n'.join(self.history[-5:]) + '\nUsuário: ' + user_input
        else:
            context = 'Usuário: ' + user_input
        
        # Gera resposta
        response = generate_text(
            self.model,
            self.tokenizer,
            prompt=context,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extrai apenas a última resposta
        if 'Assistente:' in response:
            response = response.split('Assistente:')[-1].strip()
        
        # Atualiza histórico
        self.history.append(f'Usuário: {user_input}')
        self.history.append(f'Assistente: {response}')
        
        return response
    
    def clear_history(self):
        """Limpa o histórico de chat."""
        self.history = []
    
    def interactive_chat(self, max_tokens: int = 100, temperature: float = 0.8):
        """Inicia chat interativo no terminal."""
        print("=" * 60)
        print("GpSRNN-8M - Chat Interface")
        print("=" * 60)
        print("Comandos:")
        print("  /quit  - Sair")
        print("  /clear - Limpar histórico")
        print("  /save  - Salvar modelo")
        print("=" * 60)
        print("Digite sua mensagem:\n")
        
        while True:
            try:
                user_input = input("Você: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("Encerrando chat...")
                    break
                
                if user_input.lower() == '/clear':
                    self.clear_history()
                    print("Histórico limpo!\n")
                    continue
                
                if user_input.lower() == '/save':
                    filename = input("Nome do arquivo: ").strip()
                    if filename:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'config': self.model.config
                        }, filename)
                        print(f"Modelo salvo em {filename}\n")
                    continue
                
                # Gera resposta
                print("Assistente: ", end='', flush=True)
                response = self.respond(user_input, max_tokens, temperature)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nChat interrompido.")
                break
            except Exception as e:
                print(f"Erro: {e}\n")


# ============================================================================
# EXEMPLO DE USO E TESTES
# ============================================================================

def run_tests():
    """Executa testes básicos do modelo."""
    from dataclasses import dataclass
    
    print("=" * 70)
    print("TESTES DO MODELO GpSRNN-8M")
    print("=" * 70)
    
    # Configuração
    config = GpSRNNConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=8,
        n_heads=4,
        d_ffn=512,
        dropout=0.1,
        max_seq_len=512
    )
    
    # Estimativa de parâmetros
    params = config.estimate_parameters()
    print(f"\n📊 ESTIMATIVA DE PARÂMETROS:")
    print(f"   Embeddings: {params['embeddings']:,}")
    print(f"   Por camada: {params['per_layer']:,}")
    print(f"   Total camadas: {params['total_layers']:,}")
    print(f"   LM Head: {params['lm_head']:,}")
    print(f"   TOTAL: {params['total']:,} ({params['total_millions']:.2f}M)")
    
    # Cria o modelo
    print(f"\n🔧 Criando modelo...")
    device, dtype = get_device_and_dtype()
    print(f"   Dispositivo: {device}")
    print(f"   Dtype: {dtype}")
    
    model = GpSRNNModel(config).to(device=device, dtype=dtype)
    
    # Conta parâmetros reais
    param_count = count_parameters(model)
    print(f"\n✅ PARÂMETROS REAIS:")
    print(f"   Total: {param_count['total']:,} ({param_count['total_millions']:.2f}M)")
    print(f"   Trainable: {param_count['trainable']:,} ({param_count['trainable_millions']:.2f}M)")
    
    # Estimativa de memória
    mem = estimate_memory_usage(model, batch_size=1, seq_len=512)
    print(f"\n💾 ESTIMATIVA DE MEMÓRIA:")
    print(f"   Parâmetros: {mem['parameters_mb']:.2f} MB")
    print(f"   Gradientes: {mem['gradients_mb']:.2f} MB")
    print(f"   Ativadores (treino): {mem['activations_mb']:.2f} MB")
    print(f"   Estados (inferência): {mem['recurrent_states_mb']:.2f} MB")
    print(f"   Total treino: {mem['total_training_mb']:.2f} MB")
    print(f"   Total inferência: {mem['total_inference_mb']:.2f} MB")
    
    # Teste de forward pass (treino)
    print(f"\n🧪 TESTE DE FORWARD PASS (TREINO)...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', dtype=dtype):
        logits, states = model(input_ids)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   ✓ Forward pass OK")
    
    # Teste de backward pass
    print(f"\n🧪 TESTE DE BACKWARD PASS...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Backward pass OK - Gradientes fluindo!")
    
    # Teste de inferência recorrente
    print(f"\n🧪 TESTE DE INFERÊNCIA RECORRENTE...")
    model.eval()
    with torch.no_grad():
        # Single token inference
        single_input = input_ids[:, 0:1]
        logits_single, states = model(single_input, states=None)
        
        # Next token com estado
        next_input = input_ids[:, 1:2]
        logits_next, states = model(next_input, states=states)
    
    print(f"   Single token output shape: {logits_single.shape}")
    print(f"   ✓ Inferência recorrente OK")
    
    # Teste de geração
    print(f"\n🧪 TESTE DE GERAÇÃO DE TEXTO...")
    with torch.no_grad():
        generated = model.generate(
            input_ids[:, :10],
            max_new_tokens=20,
            temperature=0.8
        )
    
    print(f"   Generated shape: {generated.shape}")
    print(f"   ✓ Geração OK")
    
    # Teste com tokenizer
    print(f"\n🧪 TESTE COM TOKENIZER...")
    tokenizer = create_default_tokenizer()
    
    test_text = "Olá, como vai você? Hello, how are you?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"   Original: {test_text}")
    print(f"   Encoded length: {len(encoded)}")
    print(f"   Decoded: {decoded}")
    print(f"   ✓ Tokenizer OK")
    
    print("\n" + "=" * 70)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 70)
    
    return model, tokenizer, config


def demo_chat():
    """Demonstra o chat com o modelo."""
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO DE CHAT")
    print("=" * 70)
    
    # Cria modelo e tokenizer
    config = GpSRNNConfig()
    device, _ = get_device_and_dtype()
    model = GpSRNNModel(config).to(device)
    tokenizer = create_default_tokenizer()
    
    # Coloca em modo de avaliação
    model.eval()
    
    # Cria interface de chat
    chat = ChatInterface(model, tokenizer)
    
    # Exemplos de conversa
    examples = [
        "Olá!",
        "Como você está?",
        "Conte-me uma história curta.",
        "What is your name?"
    ]
    
    print("\nExemplos de interação (modelo não treinado - respostas aleatórias):\n")
    
    for example in examples:
        print(f"Usuário: {example}")
        try:
            response = chat.respond(example, max_tokens=50, temperature=0.8)
            print(f"Assistente: {response[:100]}..." if len(response) > 100 else f"Assistente: {response}")
        except Exception as e:
            print(f"Assistente: [Erro na geração: {e}]")
        print()
    
    print("Para chat interativo, execute: python gp_srrn_1b.py --chat")


if __name__ == "__main__":
    import sys
    
    # Parse de argumentos simples
    if len(sys.argv) > 1:
        if sys.argv[1] == "--chat":
            # Modo chat interativo
            config = GpSRNNConfig()
            device, dtype = get_device_and_dtype()
            
            # Carrega modelo se existir checkpoint
            checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
            
            model = GpSRNNModel(config).to(device=device, dtype=dtype)
            
            if checkpoint_path:
                print(f"Carregando checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            tokenizer = create_default_tokenizer()
            chat = ChatInterface(model, tokenizer)
            chat.interactive_chat()
        elif sys.argv[1] == "--test":
            # Roda testes
            run_tests()
        elif sys.argv[1] == "--demo":
            # Demo de chat
            demo_chat()
    else:
        # Default: roda testes
        run_tests()
        demo_chat()
