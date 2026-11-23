"""
Analisis de la red social de GitHub (musae-github)
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 0. Configuracion ------------------------------------------------------------
EDGES_PATH = sys.argv[1] if len(sys.argv) > 1 else "musae_git_edges.csv"
TARGET_PATH = sys.argv[2] if len(sys.argv) > 2 else "musae_git_target.csv"

UMBRAL_EXACTO = 5000      # por encima de esto, l/D/betweenness se aproximan
N_PARES_ELL = 500         # pares aleatorios para estimar l
K_BETWEENNESS = 200       # nodos fuente para el betweenness aproximado
UMBRAL_GRADO_DIBUJO = 20  # grado minimo para entrar en el dibujo filtrado

rng = random.Random(2024)

resumen = {}   # aqui se van guardando los numeros, para el informe


def log(titulo, **valores):
    resumen[titulo] = valores
    print(f"\n=== {titulo} ===")
    for k, v in valores.items():
        print(f"  {k} = {v}")


# ---------------------------------------------------------------------------
# 1. Carga y validacion --------------------------------------------------------
edges_raw = pd.read_csv(EDGES_PATH)
print("Columnas del CSV de aristas:", edges_raw.columns.tolist())
col_a, col_b = edges_raw.columns[:2]

G = nx.from_pandas_edgelist(edges_raw, source=col_a, target=col_b)
G.remove_edges_from(list(nx.selfloop_edges(G)))

log("Carga", N=G.number_of_nodes(), L=G.number_of_edges())

if nx.is_connected(G):
    Gc = G
else:
    nodos_gigante = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(nodos_gigante).copy()
    log("Componente gigante", N_gigante=Gc.number_of_nodes(),
        fraccion=round(Gc.number_of_nodes() / G.number_of_nodes(), 4))

# ---------------------------------------------------------------------------
# 2. Caracterizacion estructural basica ----------------------------------------
grados = dict(G.degree())
k_medio = float(np.mean(list(grados.values())))
k_max = max(grados.values())
densidad = nx.density(G)

log("Grado", k_medio=round(k_medio, 2), k_max=k_max, densidad=round(densidad, 6))

valores_k = sorted(set(grados.values()))
lista_k = list(grados.values())
pk = [lista_k.count(k) / G.number_of_nodes() for k in valores_k]
# Pc(k) = P(K >= k): la acumulada complementaria. Se ve mas limpia que P(k) en
# bruto para juzgar si hay o no cola pesada (menos ruido en el extremo de k
# grande, donde cada punto de P(k) representa muy pocos nodos).
pc = np.cumsum(np.array(pk)[::-1])[::-1]

# Ajuste de ley de potencias por regresion lineal en log-log: log P(k) = -gamma*log(k) + b.
# Se calcula ANTES de dibujar para poder pintar la recta ajustada sobre los datos.
ks_fit = np.array(valores_k, dtype=float)
ps_fit = np.array(pk, dtype=float)
mask_fit = ks_fit > 0
pendiente, intercepto = np.polyfit(np.log(ks_fit[mask_fit]), np.log(ps_fit[mask_fit]), 1)
gamma = -pendiente
log("Ajuste ley de potencias (regresion lineal en log-log)", gamma=round(gamma, 2))

k_linea = np.array([ks_fit[mask_fit].min(), ks_fit[mask_fit].max()])
p_linea = np.exp(intercepto) * k_linea ** pendiente

# Cross-check opcional, mas riguroso (maxima verosimilitud) si esta instalado
# el paquete 'powerlaw'. No sustituye al ajuste de arriba, solo lo contrasta.
ajuste_mle = None
try:
    import powerlaw
    ajuste_mle = powerlaw.Fit(lista_k, discrete=True, verbose=False)
    log("Ajuste ley de potencias (maxima verosimilitud, cross-check)",
        gamma_mle=round(ajuste_mle.power_law.alpha, 2), k_min=ajuste_mle.power_law.xmin)
except ImportError:
    pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.loglog(valores_k, pk, "o", markersize=3, label="datos")
ax1.loglog(k_linea, p_linea, "-", color="red", linewidth=1.5,
           label=f"ajuste regresion: $P(k)\\sim k^{{-\\gamma}}$")
ax1.text(0.05, 0.16, f"$\\gamma$ = {gamma:.2f} (regresion, todo el rango)",
         transform=ax1.transAxes, fontsize=9, color="red",
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.85))

if ajuste_mle is not None:
    gamma_mle = ajuste_mle.power_law.alpha
    k_min_mle = ajuste_mle.power_law.xmin
    # Recta MLE anclada al dato empirico mas cercano a k_min: la normalizacion
    # (altura) se toma de P(k) en ese punto, la pendiente es la de MLE. Solo se
    # dibuja desde k_min en adelante, que es el unico rango donde el ajuste MLE
    # dice algo.
    idx_kmin = min(int(np.searchsorted(ks_fit[mask_fit], k_min_mle)), mask_fit.sum() - 1)
    k_ancla = ks_fit[mask_fit][idx_kmin]
    p_ancla = ps_fit[mask_fit][idx_kmin]
    k_linea_mle = np.array([k_ancla, ks_fit[mask_fit].max()])
    p_linea_mle = p_ancla * (k_linea_mle / k_ancla) ** (-gamma_mle)
    ax1.loglog(k_linea_mle, p_linea_mle, "--", color="blue", linewidth=1.5,
               label=f"ajuste MLE ($k\\geq{k_min_mle:.0f}$): $P(k)\\sim k^{{-\\gamma_{{MLE}}}}$")
    ax1.text(0.05, 0.04, f"$\\gamma_{{MLE}}$ = {gamma_mle:.2f} ($k_{{min}}$={k_min_mle:.0f})",
             transform=ax1.transAxes, fontsize=9, color="blue",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="blue", alpha=0.85))

ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlabel("k"); ax1.set_ylabel("P(k)")
ax1.set_title("Distribucion de grado P(k)")
ax2.loglog(valores_k, pc, "o", markersize=3, color="darkorange")
ax2.set_xlabel("k"); ax2.set_ylabel(r"$P_c(k) = P(K \geq k)$")
ax2.set_title("Acumulada complementaria")
fig.suptitle("musae-github: distribucion de grado")
fig.tight_layout()
fig.savefig("01_distribucion_grado.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Mundo pequeno: clustering y camino medio ----------------------------------
C_media = nx.average_clustering(G)
C_er_equivalente = k_medio / G.number_of_nodes()

if Gc.number_of_nodes() <= UMBRAL_EXACTO:
    ell = nx.average_shortest_path_length(Gc)
    D = nx.diameter(Gc)
    metodo_ell = "exacto"
else:
    nodos_gc = list(Gc.nodes())
    distancias = [nx.shortest_path_length(Gc, *rng.sample(nodos_gc, 2))
                  for _ in range(N_PARES_ELL)]
    ell = float(np.mean(distancias))
    D = nx.approximation.diameter(Gc)
    metodo_ell = f"muestreo de {N_PARES_ELL} pares"

ell_er_equivalente = np.log(G.number_of_nodes()) / np.log(k_medio)

log("Clustering", C_media=round(C_media, 4), C_ER_equivalente=round(C_er_equivalente, 6))
log("Camino medio y diametro", l=round(ell, 3), D=D, metodo=metodo_ell,
    l_ER_equivalente=round(ell_er_equivalente, 3))

razon_clustering = C_media / C_er_equivalente if C_er_equivalente > 0 else float("inf")
print(f"\n  Lectura rapida: C_media es ~{razon_clustering:.0f}x el valor de un ER "
      f"equivalente ({C_er_equivalente:.6f}); l={ell:.2f} frente a ln N/ln<k>="
      f"{ell_er_equivalente:.2f} de ese mismo ER. Cuanto mayor la primera razon y mas "
      "parecidos esos dos valores de l, mas clara la firma de 'mundo pequeno' "
      "(clustering alto + camino corto a la vez).")

# ---------------------------------------------------------------------------
# 4. Asortatividad, comunidades y nodos clave ----------------------------------
r = nx.degree_pearson_correlation_coefficient(G)
log("Asortatividad", r=round(r, 4))

signo = "positiva (asortativa)" if r > 0 else "negativa (disortativa)" if r < 0 else "nula"
print(f"\n  Lectura rapida: asortatividad {signo}. En redes sociales 'de trato' "
      "(Tabla 3 del PDF de contexto) suele salir r>0 -- los hubs se conectan entre "
      "si; en redes tecnologicas o biologicas suele salir r<0.")

comunidades = nx.algorithms.community.louvain_communities(G, seed=42)
Q = nx.algorithms.community.modularity(G, comunidades)
log("Comunidades", num_comunidades=len(comunidades), Q=round(Q, 4))
print("\n  Lectura rapida: una Q por encima de ~0.3 se suele tomar como indicio de "
      "estructura de comunidades real y no solo ruido -- compara con el valor obtenido.")

tamanos_comunidad = sorted((len(c) for c in comunidades), reverse=True)
plt.figure()
plt.bar(range(1, len(tamanos_comunidad) + 1), tamanos_comunidad)
plt.xlabel("Comunidad - grado k"); plt.ylabel("Numero de nodos")
plt.title(f"Tamño de las {len(comunidades)} comunidades (Louvain, Q={Q:.3f})")
plt.savefig("02_tamano_comunidades.png", dpi=150, bbox_inches="tight")
plt.close()

n_nodos = G.number_of_nodes()
k_muestra = None if n_nodos <= UMBRAL_EXACTO else K_BETWEENNESS
betweenness = nx.betweenness_centrality(G, k=k_muestra, seed=42)
top5_grado = sorted(grados.items(), key=lambda x: -x[1])[:5]
top5_btw = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
log("Nodos clave", top5_grado=top5_grado, top5_betweenness=top5_btw,
    metodo_betweenness="exacto" if k_muestra is None else f"aproximado, k={k_muestra}")

interseccion = {n for n, _ in top5_grado} & {n for n, _ in top5_btw}
print(f"\n  Lectura rapida: {len(interseccion)} de 5 nodos coinciden en el top-5 de "
      "grado y en el de betweenness. Si coinciden pocos, 'los mas populares' y "
      "'los que conectan comunidades distintas' son papeles distintos en tu red.")

plt.figure()
xs = list(grados.values())
ys = [betweenness[n] for n in grados]
plt.scatter(xs, ys, s=8, alpha=0.35, label="nodos")
plt.scatter([grados[n] for n, _ in top5_grado], [betweenness[n] for n, _ in top5_grado],
            s=45, color="red", label="top-5 grado")
plt.scatter([grados[n] for n, _ in top5_btw], [betweenness[n] for n, _ in top5_btw],
            s=45, facecolors="none", edgecolors="black", linewidths=1.3,
            label="top-5 betweenness")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("grado k"); plt.ylabel("betweenness")
plt.title("Nodos clave: grado vs. betweenness")
plt.legend(fontsize=8)
plt.savefig("03_grado_vs_betweenness.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. Contraste con la etiqueta real (web-dev / ml-dev), si esta disponible ----
color_comunidad = {}
for i, com in enumerate(comunidades):
    for nodo in com:
        color_comunidad[nodo] = i

# Subgrafo filtrado por grado y su layout, reutilizados aqui y en la Seccion 7.
nodos_dibujo = [nd for nd in G.nodes() if grados[nd] >= UMBRAL_GRADO_DIBUJO]
G_dibujo = G.subgraph(nodos_dibujo)
pos = nx.spring_layout(G_dibujo, seed=42)

etiqueta = {}
if TARGET_PATH and TARGET_PATH.lower() != "none" and Path(TARGET_PATH).exists():
    target_raw = pd.read_csv(TARGET_PATH)
    print("Columnas del CSV de etiquetas:", target_raw.columns.tolist())
    col_id, col_label = target_raw.columns[0], target_raw.columns[-1]
    etiqueta = dict(zip(target_raw[col_id], target_raw[col_label]))

    nodos_con_ambos = [nd for nd in G.nodes() if nd in etiqueta]
    comu_lista = [color_comunidad[nd] for nd in nodos_con_ambos]
    label_lista = [etiqueta[nd] for nd in nodos_con_ambos]

    try:
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(label_lista, comu_lista)
        log("Comunidades vs etiqueta real", NMI=round(nmi, 4), nodos_comparados=len(nodos_con_ambos),
            nota="0 = independientes, 1 = coinciden perfectamente")
    except ImportError:
        tabla_cruzada = pd.crosstab(pd.Series(comu_lista, name="comunidad"),
                                     pd.Series(label_lista, name="etiqueta_real"))
        print("\n=== Comunidad vs etiqueta real (sin sklearn, tabla cruzada) ===")
        print(tabla_cruzada)

    # Comparacion visual: el mismo subgrafo, coloreado por comunidad detectada
    # y por la etiqueta real, lado a lado.
    nodos_dibujo_etq = [nd for nd in G_dibujo.nodes() if nd in etiqueta]
    if nodos_dibujo_etq:
        Gs = G_dibujo.subgraph(nodos_dibujo_etq)
        pos_s = {nd: pos[nd] for nd in Gs.nodes()}
        etiquetas_unicas = sorted(set(etiqueta[nd] for nd in nodos_dibujo_etq))
        indice_etiqueta = {et: i for i, et in enumerate(etiquetas_unicas)}

        fig, (axc, axl) = plt.subplots(1, 2, figsize=(14, 7))
        nx.draw_networkx(Gs, pos_s, ax=axc, node_size=40,
                          node_color=[color_comunidad[nd] for nd in Gs.nodes()],
                          cmap=plt.cm.tab20, with_labels=False, edge_color="gray",
                          alpha=0.7, width=0.3)
        axc.set_title("Coloreado por comunidad (Louvain)"); axc.axis("off")

        nx.draw_networkx(Gs, pos_s, ax=axl, node_size=40,
                          node_color=[indice_etiqueta[etiqueta[nd]] for nd in Gs.nodes()],
                          cmap=plt.cm.Set1, with_labels=False, edge_color="gray",
                          alpha=0.7, width=0.3)
        axl.set_title("Coloreado por etiqueta real (web-dev / ml-dev)"); axl.axis("off")

        fig.suptitle("comunidad detectada vs. etiqueta real")
        fig.savefig("05_comunidad_vs_etiqueta.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
else:
    print("\n(Sin fichero de etiquetas: se omite el contraste comunidad vs web/ml-dev)")

# ---------------------------------------------------------------------------
# 6. Comparacion con modelos de referencia (ER y BA) ---------------------------
N = G.number_of_nodes()
L = G.number_of_edges()
m_ba = max(1, round(L / N))

G_er = nx.gnm_random_graph(N, L, seed=42)
G_ba = nx.barabasi_albert_graph(N, m_ba, seed=42)

# Watts-Strogatz necesita un grado par; se redondea <k> al par mas cercano.
k_ws = round(k_medio)
if k_ws % 2 != 0:
    k_ws += 1
p_ws = 0.05
G_ws = nx.watts_strogatz_graph(N, k_ws, p_ws, seed=42)


def resumen_rapido(H, nombre):
    C_H = nx.average_clustering(H)
    if H.number_of_nodes() <= UMBRAL_EXACTO:
        nodos_h = list(H.nodes())
        muestra = [nx.shortest_path_length(H, *rng.sample(nodos_h, 2))
                   for _ in range(N_PARES_ELL)]
    else:
        comp = max(nx.connected_components(H), key=len)
        Hc = H.subgraph(comp)
        nodos_h = list(Hc.nodes())
        muestra = [nx.shortest_path_length(Hc, *rng.sample(nodos_h, 2))
                   for _ in range(N_PARES_ELL)]
    ell_H = float(np.mean(muestra))
    log(f"Modelo de referencia: {nombre}", C=round(C_H, 4), l_estimado=round(ell_H, 3))
    return C_H, ell_H


C_er, ell_er = resumen_rapido(G_er, "Erdos-Renyi (mismo N, L)")
C_ws, ell_ws = resumen_rapido(G_ws, f"Watts-Strogatz (k={k_ws}, p={p_ws})")
C_ba, ell_ba = resumen_rapido(G_ba, f"Barabasi-Albert (m={m_ba})")

# Comparativa visual de C y l: mgit frente a ER, WS y BA
etiquetas_modelo = ["Real", "ER", "WS", "BA"]
colores_modelo = ["#1f77b4", "#999999", "#999999", "#999999"]
fig, (axC, axL) = plt.subplots(1, 2, figsize=(10, 4))
axC.bar(etiquetas_modelo, [C_media, C_er, C_ws, C_ba], color=colores_modelo)
axC.set_ylabel("Clustering medio (C)"); axC.set_title("Clustering")
axL.bar(etiquetas_modelo, [ell, ell_er, ell_ws, ell_ba], color=colores_modelo)
axL.set_ylabel("Camino medio (l)"); axL.set_title("Camino medio")
fig.suptitle("misae-github vs. modelos de referencia (ER, WS, BA)")
fig.tight_layout()
fig.savefig("06_comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.close(fig)


def error_relativo(valor_ref, valor_real):
    # Error relativo de una sola magnitud (C o l) frente al valor real. No se
    # combinan C y l en una unica distancia: un modelo puede acertar mucho en
    # l y muy poco en C (o al reves), y mezclarlos en un solo numero ocultaria
    # justo esa diferencia.
    return abs(valor_real - valor_ref) / valor_real


error_C = {
    "Erdos-Renyi": error_relativo(C_er, C_media),
    "Watts-Strogatz": error_relativo(C_ws, C_media),
    "Barabasi-Albert": error_relativo(C_ba, C_media),
}
error_l = {
    "Erdos-Renyi": error_relativo(ell_er, ell),
    "Watts-Strogatz": error_relativo(ell_ws, ell),
    "Barabasi-Albert": error_relativo(ell_ba, ell),
}
mas_cercano_C = min(error_C, key=error_C.get)
mas_cercano_l = min(error_l, key=error_l.get)
resumen_error_C = ", ".join(f"{nom}={val:.2f}" for nom, val in error_C.items())
resumen_error_l = ", ".join(f"{nom}={val:.2f}" for nom, val in error_l.items())
print(f"\n  Lectura rapida (clustering C): el modelo mas cercano es {mas_cercano_C} "
      f"(error relativo {resumen_error_C}; mas bajo = mas parecido).")
print(f"  Lectura rapida (camino medio l): el modelo mas cercano es {mas_cercano_l} "
      f"(error relativo {resumen_error_l}; mas bajo = mas parecido).")

if C_media > 0 and min(C_er, C_ws, C_ba) / C_media < 0.1 and max(C_er, C_ws, C_ba) / C_media < 0.5:
    print("  Aviso: los tres modelos de referencia se quedan muy por debajo de tu C real "
          "-- es una limitacion conocida de ER/WS/BA 'de libro' (sobre todo ER y BA), no "
          "un error del calculo. Merece la pena comentarlo en la conclusion.")


def distribucion_grado(H):
    gr = dict(H.degree())
    vk = sorted(set(gr.values()))
    lk = list(gr.values())
    return vk, [lk.count(k) / H.number_of_nodes() for k in vk]


vk_er, pk_er = distribucion_grado(G_er)
vk_ws, pk_ws = distribucion_grado(G_ws)
vk_ba, pk_ba = distribucion_grado(G_ba)

plt.figure()
plt.loglog(valores_k, pk, "o", markersize=3, label="Real (musae-github)")
plt.loglog(vk_er, pk_er, "s", markersize=3, label="Erdos-Renyi")
plt.loglog(vk_ws, pk_ws, "d", markersize=3, label="Watts-Strogatz")
plt.loglog(vk_ba, pk_ba, "^", markersize=3, label="Barabasi-Albert")
plt.xlabel("k"); plt.ylabel("P(k)")
plt.title("Distribucion de grado: real vs. modelos de referencia")
plt.legend()
plt.savefig("07_Pk_comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. Visualizacion final --------------------------------------------------------
# Con Gephi + ForceAtlas2 suele salir mejor a este tamano (Seccion 9 del PDF);
# aqui se deja ademas el subgrafo filtrado (calculado en la Seccion 5) para
# verlo rapido en Python.
colores = [color_comunidad[nd] for nd in G_dibujo.nodes()]
tamanos = [80 * grados[nd] / k_medio for nd in G_dibujo.nodes()]

plt.figure(figsize=(9, 9))
nx.draw_networkx(G_dibujo, pos, node_color=colores, node_size=tamanos,
                  cmap=plt.cm.tab20, with_labels=False,
                  edge_color="gray", alpha=0.7, width=0.3)
plt.axis("off")
plt.title(f"musae-github: subgrafo de nodos con k >= {UMBRAL_GRADO_DIBUJO}")
plt.savefig("04_red_filtrada.png", dpi=200, bbox_inches="tight")
plt.close()

nx.write_gexf(G, "musae_github.gexf")   # para abrir en Gephi con ForceAtlas2

# ---------------------------------------------------------------------------
# 8. Resumen final para el informe ----------------------------------------------
print("\n\n################ RESUMEN PARA EL INFORME ################")
filas_csv = []
for titulo, valores in resumen.items():
    print(f"\n{titulo}:")
    for k, v in valores.items():
        print(f"  {k}: {v}")
    if all(not isinstance(x, (dict, list, tuple)) for x in valores.values()):
        filas_csv.append({"seccion": titulo, **valores})

pd.DataFrame(filas_csv).to_csv("resumen_resultados.csv", index=False)
print("\nResumen (valores simples) tambien guardado en resumen_resultados.csv")

figuras = ["01_distribucion_grado.png", "02_tamano_comunidades.png",
           "03_grado_vs_betweenness.png", "04_red_filtrada.png"]
if etiqueta:
    figuras.append("05_comunidad_vs_etiqueta.png")
figuras += ["06_comparacion_modelos.png", "07_Pk_comparacion_modelos.png"]
print("Figuras:", ", ".join(figuras))
print("Grafo completo exportado a musae_github.gexf (para Gephi)")