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

## Passo 3 — Criar o script `vector_redis.py`

```python
import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ── 1. Conexão com Redis ──────────────────────────────────────────────────────
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensões, local, grátis

print("✅ Conectado ao Redis e modelo carregado\n")

# ── 2. Base de documentos (sua base de conhecimento) ─────────────────────────
documentos = [
    "Redis é um banco de dados em memória ultra-rápido.",
    "Redis Cluster escala horizontalmente com 16384 hash slots.",
    "Valkey é o fork open-source do Redis mantido pela Linux Foundation.",
    "CAP Theorem: nunca ter Consistência, Disponibilidade e Partição ao mesmo tempo.",
    "RDB faz snapshots periódicos; AOF persiste cada write.",
    "Sorted Sets permitem leaderboards em O(log N) com ZADD e ZRANK.",
    "Vector databases armazenam embeddings para busca semântica.",
    "RAG usa busca vetorial para dar contexto ao LLM antes de responder.",
]

# ── 3. Gerar embeddings e salvar no Redis ────────────────────────────────────
print("📥 Gerando embeddings e salvando no Redis...")
for i, doc in enumerate(documentos):
    vetor = model.encode(doc).tolist()
    r.hset(f"doc:{i}", mapping={
        "id":        str(i),
        "texto":     doc,
        "embedding": json.dumps(vetor),
    })
    print(f"  ✅ doc:{i} → {doc[:55]}...")

print(f"\n📊 {len(documentos)} documentos salvos\n")

# ── 4. Função de busca por similaridade cosseno ───────────────────────────────
def buscar_similares(query: str, top_k: int = 3) -> list:
    q_vec = model.encode(query)
    resultados = []

    for i in range(len(documentos)):
        raw = r.hget(f"doc:{i}", "embedding")
        doc_vec = np.array(json.loads(raw))

        # Similaridade cosseno: 1.0 = idêntico | 0.0 = sem relação
        score = float(
            np.dot(q_vec, doc_vec) /
            (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec))
        )
        texto = r.hget(f"doc:{i}", "texto")
        resultados.append((score, texto))

    resultados.sort(reverse=True)
    return resultados[:top_k]

# ── 5. Testar com queries ────────────────────────────────────────────────────
queries = [
    "Como o Redis escala com múltiplos servidores?",
    "Qual banco usar quando o Redis muda de licença?",
    "Como a IA busca informações relevantes antes de responder?",
]

for query in queries:
    print(f"🔍 Query: \"{query}\"")
    for score, texto in buscar_similares(query, top_k=2):
        barra = "█" * int(score * 20)
        print(f"   {barra:<20} {score:.3f} → {texto[:65]}...")
    print()

# ── 6. Inspecionar o vetor salvo ──────────────────────────────────────────────
print("🔎 Inspecionando doc:0 no Redis:")
campos = r.hgetall("doc:0")
embedding = json.loads(campos["embedding"])
print(f"   texto:      {campos['texto']}")
print(f"   dimensões:  {len(embedding)}")
print(f"   primeiros 5 valores: {[round(v, 4) for v in embedding[:5]]}")
```

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

Abra o script e adicione no final:

```python
# ── 7. Adicionar e buscar seu próprio documento ───────────────────────────────
meu_doc = "Escreva aqui qualquer texto do seu domínio."
vetor   = model.encode(meu_doc).tolist()
r.hset("doc:custom", mapping={
    "texto":     meu_doc,
    "embedding": json.dumps(vetor),
})
print(f"\n✅ Documento salvo: doc:custom")

minha_query = "Escreva uma pergunta relacionada ao texto acima"
print(f"\n🔍 Buscando: \"{minha_query}\"")
for score, texto in buscar_similares(minha_query, top_k=3):
    print(f"   [{score:.3f}] {texto[:75]}...")
```

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

```python
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# Busca aproximada — escala para 1M+ vetores em <1ms
index   = SearchIndex.from_yaml("schema.yaml")
results = index.query(VectorQuery(vetor, "embedding", num_results=10))
```

> HNSW (Hierarchical Navigable Small World) é o algoritmo padrão do Redis, Pinecone e Weaviate para busca vetorial em produção.

---

**Checkpoints:**  
✅ Redis rodando  
✅ Embeddings gerados localmente (sem API key)  
✅ Documentos salvos no Redis como HSET  
✅ Busca semântica retornando resultados relevantes  
✅ Conceito RAG compreendido na prática
