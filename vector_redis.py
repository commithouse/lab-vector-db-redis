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

# ── 7. Modo interativo ────────────────────────────────────────────────────────
print("\n💬 Modo interativo iniciado.")
print("Digite uma pergunta e pressione Enter (Ctrl+C para sair).")

try:
    while True:
        pergunta = input("\nPergunta > ").strip()
        if not pergunta:
            print("Digite uma pergunta válida ou pressione Ctrl+C para sair.")
            continue

        print(f"🔍 Query: \"{pergunta}\"")
        for score, texto in buscar_similares(pergunta, top_k=2):
            barra = "█" * int(score * 20)
            print(f"   {barra:<20} {score:.3f} → {texto[:65]}...")
except KeyboardInterrupt:
    print("\n👋 Encerrando. Até a próxima!")