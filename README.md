# 🧠 Lab: Prompt → Embedding → Redis

**MBA em Tecnologia — FIAP**  
Prof. Daniel Lemeszenski | Encontro 1 — Bônus Vector DB

---

## Objetivo

Transformar prompts de texto em **vetores numéricos (embeddings)**, armazená-los no Redis e realizar **busca por similaridade semântica** — a base de qualquer sistema RAG/IA.

---

## Conceito em 1 minuto

```
Texto: "banco de dados rápido"
         ↓  Embedding Model
Vetor: [-0.023, 0.118, 0.045, ..., 0.231]  ← 384 números
         ↓  Redis HSET
Armazenado em memória com latência <1ms
         ↓  Busca semântica
Retorna documentos com significado similar
```

Frases semanticamente próximas geram **vetores próximos** no espaço matemático.  
"Redis rápido" ≈ "IMDB veloz" → score ≈ 0.92  
"Redis rápido" ≈ "receita de bolo" → score ≈ 0.11

---

## Pré-requisitos

- Python 3.9+
- Docker instalado

---

## Passo 1 — Subir Redis via Docker

```bash
docker run -d --name redis-vector -p 6379:6379 redis:7.0-alpine
```

Verificar:

```bash
docker exec redis-vector redis-cli ping
# → PONG
```

---

## Passo 2 — Instalar dependências Python

```bash
pip install redis sentence-transformers numpy
```

> **Nota:** `sentence-transformers` baixa o modelo `all-MiniLM-L6-v2` (~90MB) no primeiro uso. Não precisa de API key — roda 100% local.

---

## Passo 3 — Usar o script `vector_redis.py`

O arquivo `vector_redis.py` já está pronto no projeto e executa este fluxo:

1. conecta no Redis e carrega o modelo de embeddings local;
2. indexa os documentos de exemplo;
3. calcula similaridade semântica para queries iniciais;
4. mostra uma inspeção de um embedding salvo;
5. entra em **modo interativo** para você digitar novas perguntas em loop.

---

## Passo 4 — Executar

```bash
python vector_redis.py
```

**Output esperado:**

```
✅ Conectado ao Redis e modelo carregado

📥 Gerando embeddings e salvando no Redis...
  ✅ doc:0 → Redis é um banco de dados em memória ultra-rápi...
  ✅ doc:1 → Redis Cluster escala horizontalmente com 16384 ...
  ...
📊 8 documentos salvos

🔍 Query: "Como o Redis escala com múltiplos servidores?"
   ████████████████     0.821 → Redis Cluster escala horizontalmente...
   ██████████           0.612 → Redis é um banco de dados...

🔍 Query: "Qual banco usar quando o Redis muda de licença?"
   ███████████████      0.756 → Valkey é o fork open-source do Redis...

🔍 Query: "Como a IA busca informações relevantes antes de responder?"
   ████████████████     0.831 → RAG usa busca vetorial para dar contexto...
   ███████████████      0.798 → Vector databases armazenam embeddings...

🔎 Inspecionando doc:0 no Redis:
   texto:      Redis é um banco de dados em memória ultra-rápido.
   dimensões:  384
   primeiros 5 valores: [-0.0234, 0.1182, -0.0451, 0.2313, 0.0892]

💬 Modo interativo iniciado.
Digite uma pergunta e pressione Enter (Ctrl+C para sair).

Pergunta > como o redis escala?
🔍 Query: "como o redis escala?"
   ████████████████     0.821 → Redis Cluster escala horizontalmente...
   ██████████           0.612 → Redis é um banco de dados...

Pergunta > qual o papel do rag?
🔍 Query: "qual o papel do rag?"
   ████████████████     0.834 → RAG usa busca vetorial para dar contexto...
   ███████████████      0.797 → Vector databases armazenam embeddings...
```

---

## Passo 5 — Verificar no Redis diretamente

```bash
# Listar todos os documentos salvos
docker exec redis-vector redis-cli KEYS "doc:*"

# Ver campos de um documento
docker exec redis-vector redis-cli HGETALL doc:0

# Contar documentos
docker exec redis-vector redis-cli DBSIZE
```

---

## Passo 6 — Adicionar seu próprio prompt

Depois das queries iniciais, o script entra automaticamente no prompt:

```text
Pergunta >
```

Nesse ponto, basta digitar novas perguntas e pressionar Enter.  
Ele continuará no loop de busca semântica até você encerrar com `Ctrl+C`.

---

## Conceitos demonstrados

| Conceito | Implementação |
|---|---|
| **Embedding** | `model.encode(texto)` → array de 384 floats |
| **Similaridade Cosseno** | `np.dot(a,b) / (norm(a) * norm(b))` → score 0–1 |
| **Redis como Vector Store** | `HSET` para salvar, `HGETALL` para recuperar |
| **Busca Semântica** | Query → vetor → comparar todos → top-k |
| **Base do RAG** | Documentos indexados → contexto → LLM responde |

---

## Limpeza

```bash
docker rm -f redis-vector
```

---

## Próximo passo — RedisVL para produção

Para milhões de vetores, use índice **HNSW** com busca em O(log N):

```bash
pip install redisvl
```

Com RedisVL, você cria índice vetorial (HNSW) e faz busca aproximada semântica
com muito mais escala e baixa latência.

> HNSW (Hierarchical Navigable Small World) é o algoritmo padrão do Redis, Pinecone e Weaviate para busca vetorial em produção.

---

**Checkpoints:**  
✅ Redis rodando  
✅ Embeddings gerados localmente (sem API key)  
✅ Documentos salvos no Redis como HSET  
✅ Busca semântica retornando resultados relevantes  
✅ Conceito RAG compreendido na prática
