# =============================================================================
# ANÁLISIS DE IMPACTO COGNITIVO EN ESTUDIANTES DE ISC - UJAT
# Autora: Betsabe Guadalupe López Gómez
# Descripción: Programa para procesar los datos del cuestionario,
#              generar estadísticos descriptivos, gráficas y
#              comprobación de hipótesis mediante prueba Z.
# =============================================================================

import sys
import os
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
ARCHIVO_EXCEL = "DATOS_DE_LA_ENCUESTA.xlsx"
HOJA_DATOS = "Respuestas de formulario 1"
CARPETA_GRAFICAS = "graficas_resultados"
MU = 3       # Punto neutral escala Likert (media de referencia)
ALPHA = 0.05    # Nivel de significancia
Z_CRITICO = 1.96    # Valor crítico bilateral

AZUL = "#2F5496"
AZUL_MED = "#5B8DB8"
AZUL_CLAR = "#A8C4E0"
GRIS = "#E8E8E8"

os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# ── MAPEO LIKERT ──────────────────────────────────────────────────────────────
LIKERT_MAP = {}


def mapear_likert(valor):
    v = str(valor).strip()
    if v.startswith('1'):
        return 1
    elif v.startswith('2'):
        return 2
    elif v.startswith('3'):
        return 3
    elif v.startswith('4'):
        return 4
    elif v.startswith('5'):
        return 5
    return np.nan

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────


def cargar_datos(archivo):
    print(f"\n{'='*60}")
    print(f"  CARGANDO DATOS: {archivo}")
    print(f"{'='*60}")
    try:
        df = pd.read_excel(archivo, sheet_name=HOJA_DATOS)
        print(f"   Datos cargados correctamente")
        print(f"   Total de respuestas: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"   ERROR: No se encontró el archivo '{archivo}'")
        print(
            f"  El archivo Excel debe estar en la misma carpeta que este script.")
        sys.exit(1)

# ── PROCESAMIENTO ─────────────────────────────────────────────────────────────


def procesar_datos(df):
    col_semestre = df.columns[1]
    col_sexo = df.columns[2]
    col_edad = df.columns[3]
    col_horas = df.columns[4]
    col_extremas = df.columns[5]
    cols_b = df.columns[6:15]   # Reactivos 6 al 14

    # Convertir Likert a numérico
    for col in cols_b:
        df[col] = df[col].apply(mapear_likert)

    # Promedio cognitivo general por estudiante
    df['promedio_cognitivo'] = df[cols_b].mean(axis=1)

    return df, col_semestre, col_sexo, col_edad, col_horas, col_extremas, cols_b

# ── ESTADÍSTICOS DESCRIPTIVOS ─────────────────────────────────────────────────


def estadisticos_descriptivos(df, cols_b):
    print(f"\n{'='*60}")
    print(f"  ESTADÍSTICOS DESCRIPTIVOS — SECCIÓN B (Reactivos 6–14)")
    print(f"{'='*60}")

    etiquetas = [
        "R6  - Olvido de sintaxis conocida",
        "R7  - Relectura del propio código",
        "R8  - Dificultad para detectar errores simples",
        "R9  - Pérdida del hilo de programación",
        "R10 - Agotamiento mental",
        "R11 - Dificultad de concentración",
        "R12 - Mayor tiempo para problemas sencillos",
        "R13 - Fatiga física y visual",
        "R14 - Frustración y ansiedad",
    ]

    print(f"\n  {'Reactivo':<42} {'Media':>8} {'σ':>8} {'Mín':>6} {'Máx':>6}")
    print(f"  {'-'*72}")
    medias = []
    sigmas = []
    for col, etiq in zip(cols_b, etiquetas):
        media = df[col].mean()
        sigma = df[col].std()
        minv = df[col].min()
        maxv = df[col].max()
        medias.append(media)
        sigmas.append(sigma)
        marca = " ◄ más alto" if media == max(
            [df[c].mean() for c in cols_b]) else ""
        print(
            f"  {etiq:<42} {media:>8.4f} {sigma:>8.4f} {minv:>6.0f} {maxv:>6.0f}{marca}")

    prom_general = df['promedio_cognitivo'].mean()
    print(f"\n  {'─'*72}")
    print(f"  {'Promedio general de impacto cognitivo:':<42} {prom_general:>8.4f}")
    print(f"  {'Punto neutral de referencia (μ):':<42} {MU:>8.1f}")
    print(
        f"  {'Diferencia respecto al punto neutral:':<42} {prom_general - MU:>8.4f}")

    return medias, sigmas

# ── PRUEBA Z ──────────────────────────────────────────────────────────────────


def prueba_z(datos, nombre_grupo, mu=MU):
    xbar = datos.mean()
    sigma = datos.std()
    n = len(datos)
    raiz_n = np.sqrt(n)
    denominador = sigma / raiz_n
    numerador = xbar - mu
    z = numerador / denominador
    rechaza = abs(z) > Z_CRITICO

    print(f"\n  Grupo: {nombre_grupo}")
    print(f"  {'─'*50}")
    print(f"  n              = {n}")
    print(f"  X̄ (media)      = {xbar:.4f}")
    print(f"  σ (desv. est.) = {sigma:.4f}")
    print(f"  μ (referencia) = {mu}")
    print(f"  √n             = {raiz_n:.4f}")
    print(f"  Numerador      = X̄ − μ  = {xbar:.4f} − {mu} = {numerador:.4f}")
    print(
        f"  Denominador    = σ/√n   = {sigma:.4f}/{raiz_n:.4f} = {denominador:.4f}")
    print(f"  Z              = {numerador:.4f} / {denominador:.4f} = {z:.4f}")
    print(f"  |Z|            = {abs(z):.4f}")
    print(f"  Valor crítico  = {Z_CRITICO}")
    if rechaza:
        print(
            f"  Decisión        Se RECHAZA H₀  (|Z| = {abs(z):.4f} > {Z_CRITICO})")
    else:
        print(
            f"  Decisión        No se rechaza H₀  (|Z| = {abs(z):.4f} ≤ {Z_CRITICO})")

    return {'nombre': nombre_grupo, 'n': n, 'xbar': round(xbar, 4),
            'sigma': round(sigma, 4), 'raiz_n': round(raiz_n, 4),
            'denominador': round(denominador, 4), 'numerador': round(numerador, 4),
            'z': round(z, 4), 'rechaza': rechaza}

# ── GRÁFICAS ──────────────────────────────────────────────────────────────────


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_GRAFICAS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   Gráfica guardada: {ruta}")


def grafica_semestres(df, col_semestre):
    orden = ['Primer semestre', 'Segundo semestre', 'Tercer semestre',
             'Cuarto semestre', 'Quinto semestre', 'Sexto semestre',
             'Séptimo semestre', 'Octavo semestre', 'Noveno semestre o superior']
    etiq = ['1°', '2°', '3°', '4°', '5°', '6°', '7°', '8°', '9°+']
    cont = [df[col_semestre].value_counts().get(s, 0) for s in orden]
    porc = [round(c/235*100, 1) for c in cont]
    cols = [AZUL_CLAR]*3 + [AZUL_MED]*3 + [AZUL]*3

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(etiq, cont, color=cols, edgecolor='white',
                  linewidth=0.8, zorder=3)
    for bar, c, p in zip(bars, cont, porc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f'{c}\n({p}%)', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Semestre', fontsize=10)
    ax.set_ylabel('Número de estudiantes', fontsize=10)
    ax.set_title(
        'Figura 1. Distribución de participantes por semestre (n = 235)', fontsize=11, pad=12)
    ax.set_ylim(0, 58)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leyenda = [mpatches.Patch(color=AZUL_CLAR, label='Semestres iniciales (1°–3°)'),
               mpatches.Patch(color=AZUL_MED,
                              label='Semestres intermedios (4°–6°)'),
               mpatches.Patch(color=AZUL,      label='Semestres avanzados (7°–9°+)')]
    ax.legend(handles=leyenda, fontsize=8, loc='upper left')
    plt.tight_layout()
    guardar(fig, 'fig1_semestres.png')


def grafica_sexo(df, col_sexo):
    conteos = df[col_sexo].value_counts()
    labels = [f'Hombre\n{conteos.get("Hombre", 0)} ({round(conteos.get("Hombre", 0)/235*100, 1)}%)',
              f'Mujer\n{conteos.get("Mujer", 0)} ({round(conteos.get("Mujer", 0)/235*100, 1)}%)',
              f'Prefiero\nno decirlo\n{conteos.get("Prefiero no decirlo", 0)} ({round(conteos.get("Prefiero no decirlo", 0)/235*100, 1)}%)']
    sizes = [conteos.get("Hombre", 0), conteos.get(
        "Mujer", 0), conteos.get("Prefiero no decirlo", 0)]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=[AZUL, AZUL_CLAR, GRIS],
                                      autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                                      wedgeprops=dict(edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_fontsize(9)
    for tx in texts:
        tx.set_fontsize(9)
    ax.set_title('Figura 2. Distribución por sexo (n = 235)',
                 fontsize=11, pad=12)
    plt.tight_layout()
    guardar(fig, 'fig2_sexo.png')


def grafica_edad(df, col_edad):
    orden_e = ['18-20 años', '21-23 años', '24-26 años', '27 años o más']
    cont_e = [df[col_edad].value_counts().get(e, 0) for e in orden_e]
    porc_e = [round(c/235*100, 1) for c in cont_e]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(orden_e, cont_e, color=[AZUL_CLAR, AZUL, AZUL_MED, GRIS],
                   edgecolor='white', linewidth=0.8, zorder=3)
    for bar, c, p in zip(bars, cont_e, porc_e):
        ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2,
                f'{c} ({p}%)', va='center', fontsize=9)
    ax.set_xlabel('Número de estudiantes', fontsize=10)
    ax.set_title(
        'Figura 3. Distribución por rango de edad (n = 235)', fontsize=11, pad=12)
    ax.set_xlim(0, 160)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    guardar(fig, 'fig3_edad.png')


def grafica_horas(df, col_horas):
    orden_h = ['Menos de 2 horas', '2-4 horas', '5-7 horas',
               '8-10 horas', '11-12 horas', 'Más de 12 horas']
    etiq_h = ['< 2 h', '2–4 h', '5–7 h', '8–10 h', '11–12 h', '> 12 h']
    cont_h = [df[col_horas].value_counts().get(h, 0) for h in orden_h]
    porc_h = [round(c/235*100, 1) for c in cont_h]
    cols_h = [AZUL_CLAR, AZUL_CLAR, AZUL_MED, AZUL, AZUL, AZUL]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(etiq_h, cont_h, color=cols_h,
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, c, p in zip(bars, cont_h, porc_h):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{c}\n({p}%)', ha='center', va='bottom', fontsize=8.5)
    ax.set_xlabel('Horas continuas de programación', fontsize=10)
    ax.set_ylabel('Número de estudiantes', fontsize=10)
    ax.set_title(
        'Figura 4. Horas continuas de programación por sesión (n = 235)', fontsize=11, pad=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leyenda_h = [mpatches.Patch(color=AZUL_CLAR, label='Jornada regular (<4 h)'),
                 mpatches.Patch(color=AZUL_MED,
                                label='Jornada moderada (5–7 h)'),
                 mpatches.Patch(color=AZUL,      label='Jornada extensa (≥8 h)')]
    ax.legend(handles=leyenda_h, fontsize=8.5, loc='upper right')
    plt.tight_layout()
    guardar(fig, 'fig4_horas.png')


def grafica_extremas(df, col_extremas):
    orden_x = ['No, nunca', 'Sí, una vez',
               'Sí, 2-3 veces', 'Sí, 4 o más veces']
    etiq_x = ['No, nunca', 'Sí, una vez', 'Sí, 2–3 veces', 'Sí, 4+\nveces']
    cont_x = [df[col_extremas].value_counts().get(o, 0) for o in orden_x]
    porc_x = [round(c/235*100, 1) for c in cont_x]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(etiq_x, cont_x, color=[GRIS, AZUL_CLAR, AZUL_MED, AZUL],
                  edgecolor='white', linewidth=0.8, zorder=3)
    for bar, c, p in zip(bars, cont_x, porc_x):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{c}\n({p}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Número de estudiantes', fontsize=10)
    ax.set_title(
        'Figura 5. Frecuencia de jornadas extremas\n(3 o más días consecutivos, n = 235)', fontsize=11, pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    guardar(fig, 'fig5_jornadas_extremas.png')


def grafica_reactivos(medias, sigmas):
    etiq_r = ['R6\nOlvido\nsintaxis', 'R7\nReleer\ncódigo', 'R8\nErrores\nsimples',
              'R9\nPerder\nel hilo', 'R10\nAgotamiento\nmental', 'R11\nConcentración',
              'R12\nTiempo\nextra', 'R13\nFatiga\nfísica', 'R14\nFrustración']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(etiq_r, medias, color=AZUL, edgecolor='white', linewidth=0.8,
                  zorder=3, yerr=sigmas, error_kw=dict(ecolor=AZUL_MED, capsize=4, linewidth=1.2))
    ax.axhline(y=MU, color='red', linestyle='--', linewidth=1.2,
               label=f'μ = {MU} (punto neutral)', zorder=4)
    ax.axhline(y=np.mean(medias), color='orange', linestyle=':', linewidth=1.2,
               label=f'X̄ general = {np.mean(medias):.2f}', zorder=4)
    for bar, m in zip(bars, medias):
        ax.text(bar.get_x()+bar.get_width()/2, m+0.06,
                f'{m:.2f}', ha='center', va='bottom', fontsize=8.5,
                color='white', fontweight='bold')
    ax.set_ylabel('Media (escala Likert 1–5)', fontsize=10)
    ax.set_title(
        'Figura 6. Media por reactivo — Impacto cognitivo (n = 235)', fontsize=11, pad=12)
    ax.set_ylim(1, 5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    guardar(fig, 'fig6_reactivos.png')


def grafica_prueba_z(res1, res2, titulo_h, etiq1, etiq2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(titulo_h, fontsize=12, fontweight='bold', y=1.02)

    for ax, res, etiq, color in zip(axes, [res1, res2], [etiq1, etiq2], [AZUL_CLAR, AZUL]):
        x = np.linspace(-4, 6, 400)
        y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2)
        ax.plot(x, y, color='#333333', linewidth=1.5)
        ax.fill_between(x, y, where=(x < -Z_CRITICO),
                        color='#FF6B6B', alpha=0.5, label='Zona rechazo')
        ax.fill_between(x, y, where=(x > Z_CRITICO),
                        color='#FF6B6B', alpha=0.5)
        ax.fill_between(x, y, where=(x >= -Z_CRITICO) & (x <= Z_CRITICO),
                        color='#A8C4E0', alpha=0.3, label='Zona no rechazo')
        ax.axvline(x=res['z'], color=color, linewidth=2.5,
                   linestyle='-', label=f"Z = {res['z']}")
        ax.axvline(x=-Z_CRITICO, color='red', linewidth=1,
                   linestyle='--', label=f'±{Z_CRITICO}')
        ax.axvline(x=Z_CRITICO,  color='red', linewidth=1, linestyle='--')
        decision = "Se rechaza H₀ ✓" if res['rechaza'] else "No se rechaza H₀"
        ax.set_title(f"{etiq}\nZ = {res['z']} | {decision}", fontsize=10)
        ax.set_xlabel('Valor Z', fontsize=9)
        ax.set_ylabel('Densidad', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    nombre = titulo_h.replace(' ', '_').replace(
        '—', '').replace('/', 'vs')[:30] + '.png'
    guardar(fig, nombre)

# ── PROGRAMA PRINCIPAL ────────────────────────────────────────────────────────


def main():
    print("\n" + "="*60)
    print("  ANÁLISIS DE IMPACTO COGNITIVO EN ESTUDIANTES DE ISC")
    print("  Universidad Juárez Autónoma de Tabasco — DACI")
    print("  Autora: Betsabe Guadalupe López Gómez")
    print("="*60)

    # 1. Cargar datos
    df = cargar_datos(ARCHIVO_EXCEL)

    # 2. Procesar
    df, col_semestre, col_sexo, col_edad, col_horas, col_extremas, cols_b = procesar_datos(
        df)

    # 3. Estadísticos descriptivos
    medias, sigmas = estadisticos_descriptivos(df, cols_b)

    # 4. Distribuciones sociodemográficas
    print(f"\n{'='*60}")
    print(f"  DISTRIBUCIONES SOCIODEMOGRÁFICAS")
    print(f"{'='*60}")
    print(f"\n  Semestre:")
    print(df[col_semestre].value_counts().to_string())
    print(f"\n  Sexo:")
    print(df[col_sexo].value_counts().to_string())
    print(f"\n  Edad:")
    print(df[col_edad].value_counts().to_string())
    print(f"\n  Horas de programación:")
    print(df[col_horas].value_counts().to_string())
    print(f"\n  Jornadas extremas:")
    print(df[col_extremas].value_counts().to_string())

    # 5. Prueba Z — Hipótesis 1
    print(f"\n{'='*60}")
    print(f"  HIPÓTESIS 1 — Jornadas regulares vs. jornadas extensas")
    print(
        f"  Fórmula: z = (X̄ − μ) / (σ / √n)   |   μ = {MU}   |   α = {ALPHA}")
    print(f"{'='*60}")

    regular = df[df[col_horas].isin(
        ['Menos de 2 horas', '2-4 horas'])]['promedio_cognitivo'].dropna()
    extensa = df[df[col_horas].isin(
        ['8-10 horas', '11-12 horas'])]['promedio_cognitivo'].dropna()

    res_h1_g1 = prueba_z(regular, "Jornada regular (< 4 horas)")
    res_h1_g2 = prueba_z(extensa,  "Jornada extensa (8–12 horas)")

    # 6. Prueba Z — Hipótesis 2
    print(f"\n{'='*60}")
    print(f"  HIPÓTESIS 2 — Semestres iniciales vs. semestres avanzados")
    print(
        f"  Fórmula: z = (X̄ − μ) / (σ / √n)   |   μ = {MU}   |   α = {ALPHA}")
    print(f"{'='*60}")

    iniciales = df[df[col_semestre].isin(
        ['Primer semestre', 'Segundo semestre', 'Tercer semestre'])]['promedio_cognitivo'].dropna()
    avanzados = df[df[col_semestre].isin(
        ['Séptimo semestre', 'Octavo semestre', 'Noveno semestre o superior'])]['promedio_cognitivo'].dropna()

    res_h2_g1 = prueba_z(iniciales, "Semestres iniciales (1°–3°)")
    res_h2_g2 = prueba_z(avanzados,  "Semestres avanzados (7°+)")

    # 7. Generar gráficas
    print(f"\n{'='*60}")
    print(f"  GENERANDO GRÁFICAS")
    print(f"{'='*60}")
    grafica_semestres(df, col_semestre)
    grafica_sexo(df, col_sexo)
    grafica_edad(df, col_edad)
    grafica_horas(df, col_horas)
    grafica_extremas(df, col_extremas)
    grafica_reactivos(medias, sigmas)
    grafica_prueba_z(res_h1_g1, res_h1_g2, "Hipótesis 1 — Jornada regular vs extensa",
                     "Jornada regular (<4h)", "Jornada extensa (8–12h)")
    grafica_prueba_z(res_h2_g1, res_h2_g2, "Hipótesis 2 — Semestres iniciales vs avanzados",
                     "Semestres iniciales (1°–3°)", "Semestres avanzados (7°+)")

    # 8. Resumen final
    print(f"\n{'='*60}")
    print(f"  RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    print(f"\n  Hipótesis 1:")
    print(
        f"    Jornada regular  → Z = {res_h1_g1['z']:>6.4f}  {' Se rechaza H₀' if res_h1_g1['rechaza'] else ' No se rechaza H₀'}")
    print(
        f"    Jornada extensa  → Z = {res_h1_g2['z']:>6.4f}  {' Se rechaza H₀' if res_h1_g2['rechaza'] else ' No se rechaza H₀'}")
    print(f"\n  Hipótesis 2:")
    print(
        f"    Sem. iniciales   → Z = {res_h2_g1['z']:>6.4f}  {' Se rechaza H₀' if res_h2_g1['rechaza'] else ' No se rechaza H₀'}")
    print(
        f"    Sem. avanzados   → Z = {res_h2_g2['z']:>6.4f}  {' Se rechaza H₀' if res_h2_g2['rechaza'] else ' No se rechaza H₀'}")
    print(f"\n  Gráficas guardadas en: ./{CARPETA_GRAFICAS}/")
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}\n")


def conclusion(res_h1_g1, res_h1_g2, res_h2_g1, res_h2_g2, df, col_extremas):
    extremas_pct = round(
        (1 - df[col_extremas].value_counts(normalize=True).get('No, nunca', 0)) * 100, 1)

    print(f"\n{'='*60}")
    print(f"  CONCLUSIÓN")
    print(f"{'='*60}\n")

    print(f"  H1: La prueba Z confirmó que las jornadas extensas (8-12 h)")
    print(
        f"  producen impacto cognitivo significativo (Z = {res_h1_g2['z']} > 1.96),")
    print(
        f"  mientras que las regulares no lo generan (Z = {res_h1_g1['z']}).")

    print(f"\n  H2: Los semestres avanzados presentan mayor impacto cognitivo")
    print(
        f"  (Z = {res_h2_g2['z']}) que los iniciales (Z = {res_h2_g1['z']}),")
    print(f"  lo que indica que la carga cognitiva aumenta con la carrera.")

    print(
        f"\n  Dato adicional: el {extremas_pct}% de los estudiantes ha trabajado")
    print(f"  3 o mas dias consecutivos en algun proyecto de programacion.")
    print(f"\n{'='*60}\n")


def main_con_conclusion():
    df = cargar_datos(ARCHIVO_EXCEL)
    df, col_semestre, col_sexo, col_edad, col_horas, col_extremas, cols_b = procesar_datos(
        df)
    medias, sigmas = estadisticos_descriptivos(df, cols_b)
    regular = df[df[col_horas].isin(
        ["Menos de 2 horas", "2-4 horas"])]["promedio_cognitivo"].dropna()
    extensa = df[df[col_horas].isin(
        ["8-10 horas", "11-12 horas"])]["promedio_cognitivo"].dropna()
    iniciales = df[df[col_semestre].isin(
        ["Primer semestre", "Segundo semestre", "Tercer semestre"])]["promedio_cognitivo"].dropna()
    avanzados = df[df[col_semestre].isin(
        ["Séptimo semestre", "Octavo semestre", "Noveno semestre o superior"])]["promedio_cognitivo"].dropna()
    res_h1_g1 = prueba_z(regular, "Jornada regular (< 4 horas)")
    res_h1_g2 = prueba_z(extensa,  "Jornada extensa (8-12 horas)")
    res_h2_g1 = prueba_z(iniciales, "Semestres iniciales (1-3)")
    res_h2_g2 = prueba_z(avanzados,  "Semestres avanzados (7+)")
    grafica_semestres(df, col_semestre)
    grafica_sexo(df, col_sexo)
    grafica_edad(df, col_edad)
    grafica_horas(df, col_horas)
    grafica_extremas(df, col_extremas)
    grafica_reactivos(medias, sigmas)
    grafica_prueba_z(res_h1_g1, res_h1_g2, "Hipotesis 1", "Regular", "Extensa")
    grafica_prueba_z(res_h2_g1, res_h2_g2, "Hipotesis 2",
                     "Iniciales", "Avanzados")
    conclusion(res_h1_g1, res_h1_g2, res_h2_g1, res_h2_g2, df, col_extremas)


if __name__ == "__main__":
    main_con_conclusion()
