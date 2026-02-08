
import numpy as np
import matplotlib.pyplot as plt

# ==================================
# ETAPA 1 — Geração de dados
# ==================================

num_usuarios = 1000

num_visitas = np.random.randint(1, 16, size=num_usuarios)
tempo = np.random.normal(loc=10, scale=2, size=num_usuarios) + (num_visitas * 0.7)
itensc = np.random.randint(0, 30, size=num_usuarios)

valorc = itensc * np.random.normal(loc=50, scale=2, size=num_usuarios)

idade_us = np.random.randint(18, 66, size=num_usuarios)
renda_es = np.random.normal(loc=3000, scale=1000, size=num_usuarios)

valor_compra = valorc + (renda_es / 10)
valor_compra[itensc == 0] = 0


# ==================================
# ETAPA 2 — Estrutura do dataset
# ==================================

dados_ecommerce = np.column_stack((
    num_visitas.round(2),
    tempo.round(2),
    itensc.round(2),
    valor_compra.round(2),
    idade_us.round(2),
    renda_es.round(2)
))

usuarios_sc = dados_ecommerce[:, 2] == 0
quantida = np.sum(usuarios_sc)

p_quem_compra = (np.sum(dados_ecommerce[:, 3] > 0) / num_usuarios) * 100


# ==================================
# Separação de colunas
# ==================================

visitas_col = dados_ecommerce[:, 0]
tempo_col   = dados_ecommerce[:, 1]
itens_col   = dados_ecommerce[:, 2]
valor_col   = dados_ecommerce[:, 3]
idade_col   = dados_ecommerce[:, 4]
renda_col   = dados_ecommerce[:, 5]


# ==================================
# ETAPA 3 — Estatística
# ==================================

media_visitas = np.mean(visitas_col)
media_tempo   = np.mean(tempo_col)
media_itens   = np.mean(itens_col)
media_valor   = np.mean(valor_col)

mediana_visitas = np.median(visitas_col)
mediana_tempo   = np.median(tempo_col)
mediana_itens   = np.median(itens_col)
mediana_valor   = np.median(valor_col)

max_visitas = np.max(visitas_col)
max_tempo   = np.max(tempo_col)
max_itens   = np.max(itens_col)
max_valor   = np.max(valor_col)

std_visitas = np.std(visitas_col)
std_tempo   = np.std(tempo_col)
std_itens   = np.std(itens_col)
std_valor   = np.std(valor_col)

print("\n--- Estatísticas ---")
print(f"Média valor: {media_valor:.2f}")
print(f"Mediana valor: {mediana_valor:.2f}")
print(f"Desvio padrão valor: {std_valor:.2f}")
print(f"Maior valor: {max_valor:.2f}")

print(f"\nMédia visitas: {media_visitas:.2f}")
print(f"Média tempo: {media_tempo:.2f}")
print(f"Média itens: {media_itens:.2f}")


# ==================================
# ETAPA 4 — Visualização
# ==================================

plt.figure(figsize=(10, 5))
plt.hist(valor_col, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
plt.title("Distribuição do Valor de Compra", fontsize=14)
plt.xlabel("Valor (R$)")
plt.ylabel("Frequência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(tempo_col, bins=30, color='lightgreen', edgecolor='black', alpha=0.8)
plt.title("Distribuição do Tempo no Site", fontsize=14)
plt.xlabel("Tempo (minutos)")
plt.ylabel("Frequência")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(tempo_col, valor_col, alpha=0.6)
plt.title("Tempo no Site vs Valor Gasto", fontsize=14)
plt.xlabel("Tempo no site")
plt.ylabel("Valor gasto")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================
# ETAPA 5 — Correlação
# ==================================

pxv = np.corrcoef(visitas_col, valor_col)[0, 1]
pxr = np.corrcoef(renda_col, valor_col)[0, 1]
pxi = np.corrcoef(idade_col, valor_col)[0, 1]

print("\n--- Correlações ---")
print(f"Visitas x valor: {pxv:.2f}")
print(f"Renda x valor: {pxr:.2f}")
print(f"Idade x valor: {pxi:.2f}")