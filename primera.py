import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import g # Gravedad terrestre
from scipy.integrate import odeint # Para resolver ecuaciones diferenciales, útil en animaciones complejas

# -------------------- Configuración de la Página (¡DEBE SER LA PRIMERA!) --------------------
st.set_page_config(layout="wide", page_title="Simulaciones de Física: Impulso y Cantidad de Movimiento")

# --- CSS Personalizado para la Interfaz Creativa (Va DESPUÉS de set_page_config) ---
background_image_url = "https://i.postimg.cc/CMBSnW4f/eee.jpg" # ¡CAMBIA ESTA URL por una tuya si la prueba funciona!

# Importante: Asegúrate de que no haya espacios o caracteres invisibles antes de '<style>'
st.markdown(
    f"""
<style>
/* Estilos para el fondo de la aplicación */
.stApp {{
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Estilos para el contenedor principal del contenido */
.stSidebar {{
    background-color: rgba(0, 0, 0, 0.8);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

/* Estilos para la barra lateral */
.stSidebar {{
    background-color: rgba(240, 240, 240, 0.9);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

/* Estilos para los encabezados (H1 a H6) - ROJO */
h1[data-testid="stAppViewTitle"],
h2[data-testid^="stHeader"],
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6
{{
    color: red !important;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    font-size: 2.2em;
    font-weight: bold;
}}

/* Estilos para el texto de párrafo, listas, span y divs generales - NARANJA */
.stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {{
    color: white !important;
    font-size: 1.1em;
    font-weight: 300;
}}

/* Para el valor numérico en st.metric */
.st-bd {{
    color: white !important;
}}

/* Para el texto dentro de st.info, st.warning, st.error boxes */
.st-dg {{
    color: #555555 !important;
    font-weight: 500;
}}

/* Para los labels de los widgets (sliders, inputs, selectbox, radio) - AHORA TAMBIÉN EN ROJO */
.stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label {{
    font-size: 1.15em;
    font-weight: 600;
    color: white !important; /* ¡CAMBIADO A ROJO! */
}}

/* Para el texto de las opciones de radio buttons y selectboxes */
div[data-testid="stRadio"] label span,
div[data-testid="stSelectbox"] div[role="button"] span {{
    color: orange !important; /* Las opciones mismas, siguen en naranja */
}}

/* Estilos para el texto dentro de los botones */
.stButton > button {{
    font-size: 1.1em;
    font-weight: 600;
    color: #333333 !important;
}}

/* Asegurar que el texto dentro de los "streamlit.latex" también se vea afectado */
.st-be.st-bb, .st-bh {{
    font-size: 1.1em !important;
    font-weight: 500 !important;
    color: orange !important;
}}

</style>
    """,
    unsafe_allow_html=True
)


def calcular_impulso_fuerza(parametro_entrada, valor_entrada, tiempo=None):
    """Calcula impulso o fuerza promedio."""
    if parametro_entrada == "impulso":
        # Se tiene fuerza y tiempo, calcular impulso
        impulso = valor_entrada * tiempo
        return impulso, f"Impulso: {impulso:.2f} Ns"
    elif parametro_entrada == "fuerza_promedio":    
        # Se tiene impulso y tiempo, calcular fuerza promedio
        fuerza = valor_entrada / tiempo
        return fuerza, f"Fuerza promedio: {fuerza:.2f} N"

def simular_colision_1d(m1, v1_inicial, m2, v2_inicial, e):
    """
    Simula una colisión unidimensional (elástica, inelástica o parcialmente elástica).
    e: coeficiente de restitución (0 para inelástica, 1 para elástica)
    """
    # Conservación de la cantidad de movimiento: m1*v1i + m2*v2i = m1*v1f + m2*v2f
    # Coeficiente de restitución: e = -(v1f - v2f) / (v1i - v2i) => v1f - v2f = -e * (v1i - v2i)

    v1_final = ((m1 - e * m2) * v1_inicial + (1 + e) * m2 * v2_inicial) / (m1 + m2)
    v2_final = ((1 + e) * m1 * v1_inicial + (m2 - e * m1) * v2_inicial) / (m1 + m2)
    
    return v1_final, v2_final

def simular_colision_2d(m1, v1_inicial_x, v1_inicial_y, m2, v2_inicial_x, v2_inicial_y, e, angulo_impacto_deg):
    """
    Simula una colisión 2D entre dos partículas.
    Simplificado: asume que el impacto ocurre a lo largo de un eje definido por angulo_impacto_deg.
    Para una colisión más real, necesitarías la posición de los centros y el radio de las partículas.
    """
    angulo_impacto_rad = np.deg2rad(angulo_impacto_deg)

    # Transformar velocidades a un sistema de coordenadas donde el eje x' está a lo largo de la línea de impacto
    v1i_normal = v1_inicial_x * np.cos(angulo_impacto_rad) + v1_inicial_y * np.sin(angulo_impacto_rad)
    v1i_tangencial = -v1_inicial_x * np.sin(angulo_impacto_rad) + v1_inicial_y * np.cos(angulo_impacto_rad)
    v2i_normal = v2_inicial_x * np.cos(angulo_impacto_rad) + v2_inicial_y * np.sin(angulo_impacto_rad)
    v2i_tangencial = -v2_inicial_x * np.sin(angulo_impacto_rad) + v2_inicial_y * np.cos(angulo_impacto_rad)

    # Aplicar colisión 1D en el eje normal
    v1f_normal, v2f_normal = simular_colision_1d(m1, v1i_normal, m2, v2i_normal, e)

    # Las velocidades tangenciales se conservan
    v1f_tangencial = v1i_tangencial
    v2f_tangencial = v2i_tangencial

    # Transformar velocidades finales de vuelta al sistema de coordenadas original (x, y)
    v1_final_x = v1f_normal * np.cos(angulo_impacto_rad) - v1f_tangencial * np.sin(angulo_impacto_rad)
    v1_final_y = v1f_normal * np.sin(angulo_impacto_rad) + v1f_tangencial * np.cos(angulo_impacto_rad)
    v2_final_x = v2f_normal * np.cos(angulo_impacto_rad) - v2f_tangencial * np.sin(angulo_impacto_rad)
    v2_final_y = v2f_normal * np.sin(angulo_impacto_rad) + v2f_tangencial * np.cos(angulo_impacto_rad)

    return (v1_final_x, v1_final_y), (v2_final_x, v2_final_y)

def calcular_v_sistema_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial):
    """
    Calcula la velocidad del sistema bala+bloque justo después del impacto.
    Asume una colisión perfectamente inelástica.
    """
    # Conservación de la Cantidad de Movimiento (colisión inelástica)
    return (masa_bala * velocidad_bala_inicial) / (masa_bala + masa_bloque)

def calcular_h_max_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial):
    """
    Calcula la altura máxima alcanzada por el sistema bala+bloque.
    """
    v_sistema = calcular_v_sistema_pendulo(masa_bloque, masa_bala, velocidad_bala_inicial)
    # Conservación de la Energía Mecánica (sistema bala+bloque asciende)
    h_max = (v_sistema**2) / (2 * g)
    return h_max

def simular_flecha_saco(m_flecha, v_flecha_inicial, m_saco, mu_k):
    """
    Simula una flecha que se incrusta en un saco y lo desplaza hasta detenerse.
    """
    # 1. Colisión perfectamente inelástica (flecha se incrusta en saco)
    v_sistema_inicial = (m_flecha * v_flecha_inicial) / (m_flecha + m_saco)

    # 2. Movimiento del sistema con fricción
    m_total = m_flecha + m_saco
    F_friccion = mu_k * m_total * g # Fuerza de fricción cinética
    a_friccion = -F_friccion / m_total # Aceleración debido a la fricción (negativa)

    # 3. Distancia recorrida hasta detenerse (v_final^2 = v_inicial^2 + 2*a*d)
    if a_friccion == 0: # Evitar división por cero si no hay fricción
        distancia_detencion = float('inf') # Se movería indefinidamente
    else:
        distancia_detencion = - (v_sistema_inicial**2) / (2 * a_friccion)

    return v_sistema_inicial, F_friccion, distancia_detencion

def simular_caida_plano_impacto(m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto):
    """
    Simula un objeto deslizándose por un plano inclinado y luego impactando el suelo.
    """
    angulo_plano_rad = np.deg2rad(angulo_plano_deg)

    # 1. Movimiento en el plano inclinado
    g_paralelo = g * np.sin(angulo_plano_rad)
    g_perpendicular = g * np.cos(angulo_plano_rad)
    F_normal = m_obj * g_perpendicular
    F_friccion_plano = mu_k_plano * F_normal
    a_plano = g_paralelo - (F_friccion_plano / m_obj)

    if a_plano < 0: # Si la fricción es muy alta y no se mueve
        st.warning("El objeto no se moverá por el plano inclinado debido a la alta fricción.")
        return 0, 0, 0, 0, 0, 0, 0

    longitud_plano = altura_inicial / np.sin(angulo_plano_rad)
    v_final_plano = np.sqrt(2 * a_plano * longitud_plano)

    # 2. Impacto con el suelo (horizontal)
    vx_impacto = v_final_plano * np.cos(angulo_plano_rad)
    vy_impacto = -v_final_plano * np.sin(angulo_plano_rad)

    # Velocidad vertical de rebote (solo afecta la componente Y)
    vy_rebote = -e_impacto * vy_impacto

    # 3. Trayectoria después del rebote (tiro parabólico)
    altura_max_rebote = (vy_rebote**2) / (2 * g)
    tiempo_vuelo_rebote = (2 * vy_rebote) / g

    distancia_horizontal_rebote = vx_impacto * tiempo_vuelo_rebote

    return (a_plano, v_final_plano, vx_impacto, vy_impacto,
            vy_rebote, altura_max_rebote, distancia_horizontal_rebote)

# -------------------- Funciones de Visualización (Plotly) --------------------

def plot_colision_1d_animacion(m1, v1_inicial, m2, v2_inicial, e):
    v1_f, v2_f = simular_colision_1d(m1, v1_inicial, m2, v2_inicial, e)

    pos_inicial_1 = -5
    pos_inicial_2 = 5
    radio_1 = m1**0.3 * 0.5 # Tamaño visual basado en masa
    radio_2 = m2**0.3 * 0.5

    num_frames = 100
    t = np.linspace(0, 2, num_frames) # Tiempo total de la animación

    frames = []
    for k in range(num_frames):
        # Antes de la colisión (asumiendo que colisionan alrededor de t=1)
        if t[k] < 1:
            x1 = pos_inicial_1 + v1_inicial * t[k]
            x2 = pos_inicial_2 + v2_inicial * t[k]
        # Después de la colisión (simplificado, asume que la colisión es instantánea en t=1)
        else:
            x1 = pos_inicial_1 + v1_inicial * 1 + v1_f * (t[k] - 1)
            x2 = pos_inicial_2 + v2_inicial * 1 + v2_f * (t[k] - 1)

        # Simplificación para evitar superposición visual en el momento de impacto
        if abs(x1 - x2) < (radio_1 + radio_2) * 0.8 and t[k] < 1.05:
            pass
        else:
            if t[k] < 1:
                x1 = pos_inicial_1 + v1_inicial * t[k]
                x2 = pos_inicial_2 + v2_inicial * t[k]
            else:
                x1 = pos_inicial_1 + v1_inicial * 1 + v1_f * (t[k] - 1)
                x2 = pos_inicial_2 + v2_inicial * 1 + v2_f * (t[k] - 1)

        frame_data = [
            go.Scatter(x=[x1], y=[0], mode='markers', marker=dict(size=radio_1*20, color='blue'), name=f'Objeto 1 (Masa: {m1} kg)'),
            go.Scatter(x=[x2], y=[0], mode='markers', marker=dict(size=radio_2*20, color='red'), name=f'Objeto 2 (Masa: {m2} kg)')
        ]
        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig = go.Figure(
        data=[
            go.Scatter(x=[pos_inicial_1], y=[0], mode='markers', marker=dict(size=radio_1*20, color='blue')),
            go.Scatter(x=[pos_inicial_2], y=[0], mode='markers', marker=dict(size=radio_2*20, color='red'))
        ],
        layout=go.Layout(
            xaxis=dict(range=[-10, 10], autorange=False, zeroline=False),
            yaxis=dict(range=[-1, 1], autorange=False, showgrid=False, zeroline=False, showticklabels=False),
            title='Simulación de Colisión 1D',
            updatemenus=[dict(type='buttons', buttons=[dict(label='Play',
                                                             method='animate',
                                                             args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])]
        ),
        frames=frames
    )
    return fig

def plot_colision_2d_trayectorias(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto_deg):
    """
    Genera una visualización 2D de trayectorias antes y después de la colisión.
    """
    (v1fx, v1fy), (v2fx, v2fy) = simular_colision_2d(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto_deg)

    # Puntos de partida para las trayectorias (arbitrarios para visualización)
    p1_start = [-10, 0]
    p2_start = [10, 0]

    # Punto de colisión (arbitrario, por ejemplo, el origen)
    colision_point = [0, 0]

    # Calcular puntos de la trayectoria antes de la colisión
    t_pre_colision = np.linspace(-1, 0, 50)
    x1_pre = [p1_start[0] + v1_ix * t for t in t_pre_colision]
    y1_pre = [p1_start[1] + v1_iy * t for t in t_pre_colision]
    x2_pre = [p2_start[0] + v2_ix * t for t in t_pre_colision]
    y2_pre = [p2_start[1] + v2_iy * t for t in t_pre_colision]

    # Calcular puntos de la trayectoria después de la colisión
    t_post_colision = np.linspace(0, 1, 50)
    x1_post = [colision_point[0] + v1fx * t for t in t_post_colision]
    y1_post = [colision_point[1] + v1fy * t for t in t_post_colision]
    x2_post = [colision_point[0] + v2fx * t for t in t_post_colision]
    y2_post = [colision_point[1] + v2fy * t for t in t_post_colision]

    fig = go.Figure()

    # Trayectorias antes
    fig.add_trace(go.Scatter(x=x1_pre, y=y1_pre, mode='lines', name='Objeto 1 (Antes)', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=x2_pre, y=y2_pre, mode='lines', name='Objeto 2 (Antes)', line=dict(color='red', dash='dot')))

    # Objetos en el momento de la colisión
    fig.add_trace(go.Scatter(x=[colision_point[0]], y=[colision_point[1]], mode='markers',
                             marker=dict(size=m1*10, color='blue', symbol='circle'), name='Objeto 1 (Colisión)'))
    fig.add_trace(go.Scatter(x=[colision_point[0]], y=[colision_point[1]], mode='markers',
                             marker=dict(size=m2*10, color='red', symbol='circle'), name='Objeto 2 (Colisión)'))

    # Trayectorias después
    fig.add_trace(go.Scatter(x=x1_post, y=y1_post, mode='lines', name='Objeto 1 (Después)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x2_post, y=y2_post, mode='lines', name='Objeto 2 (Después)', line=dict(color='red')))

    fig.update_layout(title='Simulación de Colisión 2D con Trayectorias',
                      xaxis_title='Posición X',
                      yaxis_title='Posición Y',
                      xaxis_range=[-12, 12], yaxis_range=[-10, 10],
                      showlegend=True,
                      # Corrección para el aspect ratio
                      yaxis=dict(
                          scaleanchor="x",
                          scaleratio=1
                      ),
                      hovermode="closest")
    return fig

# Definición de la ecuación diferencial para el péndulo
def pendulo_eq(y, t, L):
    theta, omega = y
    dydt = [omega, -(g / L) * np.sin(theta)]
    return dydt

def plot_pendulo_balistico_animacion(masa_bala, masa_caja, velocidad_bala_inicial, largo_pendulo_vis):
    """
    Genera una animación del péndulo balístico con la bala impactando la caja.
    Muestra la bala, la caja y la cuerda.
    """
    # Cálculos preliminares
    v_sistema = calcular_v_sistema_pendulo(masa_caja, masa_bala, velocidad_bala_inicial)

    if largo_pendulo_vis <= 0:
        st.error("El largo del péndulo debe ser mayor que cero para la animación.")
        return go.Figure()

    # --- Configuración de Tiempos y Frames ---
    # Tiempo para que la bala alcance la caja
    distancia_inicial_bala = largo_pendulo_vis * 1.5 # La bala empieza a 1.5 veces el largo del péndulo a la izquierda
    tiempo_pre_impacto = distancia_inicial_bala / velocidad_bala_inicial if velocidad_bala_inicial > 0 else 0.01

    # Tiempo de oscilación del péndulo (se usa la aproximación del péndulo simple para el rango de tiempo)
    periodo_pendulo_simple = 2 * np.pi * np.sqrt(largo_pendulo_vis / g)
    tiempo_oscilacion = periodo_pendulo_simple * 1.5 # Para ver un poco más de una oscilación

    num_frames = 200 # Número de fotogramas para la animación
    tiempo_total_simulacion = tiempo_pre_impacto + tiempo_oscilacion
    t_values = np.linspace(0, tiempo_total_simulacion, num_frames)

    # --- Arrays para almacenar posiciones ---
    pos_bala_x = np.zeros(num_frames)
    pos_bala_y = np.zeros(num_frames)
    pos_caja_x = np.zeros(num_frames)
    pos_caja_y = np.zeros(num_frames)

    # --- FASE 1: Movimiento de la bala antes del impacto ---
    pos_caja_equilibrio_y = -largo_pendulo_vis # El péndulo cuelga hacia abajo desde (0,0)

    for i, t in enumerate(t_values):
        if t <= tiempo_pre_impacto:
            # Bala moviéndose hacia la caja
            pos_bala_x[i] = -distancia_inicial_bala + velocidad_bala_inicial * t
            pos_bala_y[i] = pos_caja_equilibrio_y + 0.1 # Pequeño offset para que la bala no esté exactamente en el centro de la caja

            # Caja estática
            pos_caja_x[i] = 0
            pos_caja_y[i] = pos_caja_equilibrio_y
        else:
            # --- FASE 2: Movimiento del péndulo (bala + caja) ---
            t_post_impacto = t - tiempo_pre_impacto

            y0_pendulo = [0, v_sistema / largo_pendulo_vis]

            # Resolver la ecuación diferencial del péndulo desde el tiempo de impacto
            t_solve = np.linspace(0, t_post_impacto, max(2, int((i - (num_frames * tiempo_pre_impacto // tiempo_total_simulacion)))))
            sol = odeint(pendulo_eq, y0_pendulo, t_solve, args=(largo_pendulo_vis,))

            current_theta = sol[-1, 0]

            # Posiciones del péndulo (bala + caja) en función del ángulo y largo
            pos_x_current = largo_pendulo_vis * np.sin(current_theta)
            pos_y_current = -largo_pendulo_vis * np.cos(current_theta)

            pos_bala_x[i] = pos_x_current # Bala y caja se mueven juntos
            pos_bala_y[i] = pos_y_current

            pos_caja_x[i] = pos_x_current
            pos_caja_y[i] = pos_y_current

    # --- Creación del Gráfico Plotly ---
    fig = go.Figure(
        data=[
            # Bala (solo visible antes del impacto y luego 'fusionada' con la caja)
            go.Scatter(x=[pos_bala_x[0]], y=[pos_bala_y[0]], mode='markers',
                       marker=dict(size=masa_bala*200 + 5, color='orange', symbol='circle'),
                       name='Bala',
                       showlegend=True),
            # Caja / Sistema (bala+caja después del impacto)
            go.Scatter(x=[pos_caja_x[0]], y=[pos_caja_y[0]], mode='markers',
                       marker=dict(size=masa_caja*30 + 30, color='brown', symbol='square'), # Simbolo de cuadrado para caja
                       name='Caja / Sistema',
                       showlegend=True),
            # Cuerda del péndulo (desde el pivote 0,0 al centro de la caja)
            go.Scatter(x=[0, pos_caja_x[0]], y=[0, pos_caja_y[0]], mode='lines',
                       line=dict(color='black', width=2),
                       name='Cuerda',
                       showlegend=False)
        ],
        layout=go.Layout(
            xaxis=dict(range=[-largo_pendulo_vis * 2, largo_pendulo_vis * 2], zeroline=True, title="Posición X (m)"),
            yaxis=dict(range=[-largo_pendulo_vis * 1.5, 0.5], zeroline=True, title="Posición Y (m)",
                       scaleanchor="x", scaleratio=1), # Corrección del aspect ratio
            title='Animación de Péndulo Balístico',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True},
                                                         'fromcurrent': True,
                                                         'mode': 'immediate'}])],
                              x=0.05, y=1.05, xanchor='left', yanchor='bottom' # Posición del botón Play
                            )
                        ],
            showlegend=True,
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
        ),
        frames=[go.Frame(
            data=[
                # Datos de la bala (se 'esconde' después del impacto)
                go.Scatter(x=[pos_bala_x[k]] if t_values[k] <= tiempo_pre_impacto + 0.01 else [pos_caja_x[k]], # Fusiona la bala con la caja visualmente
                           y=[pos_bala_y[k]] if t_values[k] <= tiempo_pre_impacto + 0.01 else [pos_caja_y[k]],
                           mode='markers',
                           marker=dict(size=masa_bala*200 + 5, color='orange', symbol='circle')
                          ),
                # Datos de la caja/sistema
                go.Scatter(x=[pos_caja_x[k]], y=[pos_caja_y[k]], mode='markers',
                           marker=dict(size=masa_caja*30 + 30, color='brown', symbol='square')
                          ),
                # Datos de la cuerda
                go.Scatter(x=[0, pos_caja_x[k]], y=[0, pos_caja_y[k]], mode='lines',
                           line=dict(color='black', width=2))
            ],
            name=str(k)
        ) for k in range(num_frames)]
    )

    return fig

def plot_flecha_saco_animacion(m_flecha, v_flecha_inicial, m_saco, mu_k):
    """Genera una animación de la flecha incrustándose en el saco y moviéndose."""
    v_sistema_inicial = (m_flecha * v_flecha_inicial) / (m_flecha + m_saco)
    m_total = m_flecha + m_saco
    F_friccion = mu_k * m_total * g
    a_friccion = -F_friccion / m_total if m_total > 0 else 0
    
    if a_friccion == 0:
        tiempo_detencion = 5 # Tiempo arbitrario si no hay fricción (para que se mueva)
        distancia_detencion = v_sistema_inicial * tiempo_detencion
    else:
        tiempo_detencion = abs(v_sistema_inicial / a_friccion)
        # Asegurar que la distancia sea calculada correctamente, incluso si el tiempo_detencion es 0
        distancia_detencion = v_sistema_inicial * tiempo_detencion + 0.5 * a_friccion * tiempo_detencion**2


    num_frames = 150
    tiempo_animacion = np.linspace(0, tiempo_detencion * 1.2, num_frames) # Un poco más para ver la detención

    saco_width = 1.5
    saco_height = 1.0
    flecha_length = 0.8
    flecha_height = 0.1

    frames = []
    for t in tiempo_animacion:
        # Fases del movimiento: bala se mueve, luego impacta y se mueven juntos
        # Usamos un tiempo_impacto_visual para la transición de la animación
        tiempo_impacto_visual = tiempo_animacion[-1] * 0.2 # Impacto visual a 20% del tiempo total de animación

        if t < tiempo_impacto_visual and v_flecha_inicial > 0:
            # Bala moviéndose hacia el saco
            x_flecha = -flecha_length * 2 + v_flecha_inicial * t * (flecha_length * 2) / (tiempo_impacto_visual * v_flecha_inicial) # Ajuste para que llegue al saco
            x_saco = 0 # Saco estático
        else:
            # Sistema flecha-saco moviéndose juntos después del impacto
            tiempo_post_impacto = max(0, t - tiempo_impacto_visual)
            x_sistema = v_sistema_inicial * tiempo_post_impacto + 0.5 * a_friccion * tiempo_post_impacto**2
            x_flecha = x_sistema - flecha_length / 2 # La flecha está incrustada
            x_saco = x_sistema

        frame_data = [
            go.Scatter(x=[x_flecha + flecha_length / 2], y=[saco_height/2 + flecha_height/2], # Posición de la flecha
                       mode='markers', marker=dict(size=m_flecha*300 + 10, color='gray', symbol='arrow-right')),
            go.Scatter(x=[x_saco + saco_width / 2], y=[saco_height / 2], # Posición del saco
                       mode='markers', marker=dict(size=m_saco*50 + 50, color='tan', symbol='square'))
        ]
        frames.append(go.Frame(data=frame_data, name=f'{t:.2f}'))

    fig = go.Figure(
        data=[
            go.Scatter(x=[-flecha_length/2], y=[saco_height/2 + flecha_height/2], # Posición inicial de la flecha
                       mode='markers', marker=dict(size=m_flecha*300 + 10, color='gray', symbol='arrow-right')),
            go.Scatter(x=[saco_width/2], y=[saco_height/2], # Posición inicial del saco
                       mode='markers', marker=dict(size=m_saco*50 + 50, color='tan', symbol='square'))
        ],
        layout=go.Layout(
            xaxis=dict(range=[-flecha_length * 2 -1, distancia_detencion + saco_width + 1], autorange=False, title="Posición X (m)"),
            yaxis=dict(range=[-0.5, saco_height + 0.5], autorange=False, showgrid=False, zeroline=True, showticklabels=False,
                       scaleanchor="x", scaleratio=1), # Corrección del aspect ratio
            title='Animación: Flecha se Incrusta en Saco',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])],
            shapes=[
                # Saco estático de fondo
                dict(type="rect", x0=0, y0=0, x1=saco_width, y1=saco_height,
                     fillcolor="tan", opacity=0.8, line_width=0, layer="below")
            ],
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
        ),
        frames=frames
    )
    return fig

def plot_caida_plano_impacto_animacion(m_obj, altura_inicial, angulo_plano_deg, mu_k_plano, e_impacto):
    """Animación de la caída por un plano inclinado y el rebote, mostrando la trayectoria completa."""
    angulo_plano_rad = np.deg2rad(angulo_plano_deg)
    longitud_plano = altura_inicial / np.sin(angulo_plano_rad)

    g_paralelo = g * np.sin(angulo_plano_rad)
    g_perpendicular = g * np.cos(angulo_plano_rad)
    F_normal = m_obj * g_perpendicular
    F_friccion_plano = mu_k_plano * F_normal
    a_plano = g_paralelo - (F_friccion_plano / m_obj)

    if a_plano <= 0:
        st.warning("El objeto no se moverá por el plano inclinado debido a la alta fricción.")
        return go.Figure() # No animation if no movement

    tiempo_bajada = np.sqrt(2 * longitud_plano / a_plano) if a_plano > 0 else 0
    v_final_plano = a_plano * tiempo_bajada

    vx_impacto = v_final_plano * np.cos(angulo_plano_rad)
    vy_impacto = -v_final_plano * np.sin(angulo_plano_rad)
    vy_rebote = -e_impacto * vy_impacto
    tiempo_vuelo_rebote = (2 * vy_rebote) / g if g > 0 else 0
    altura_max_rebote = (vy_rebote**2) / (2 * g)
    distancia_horizontal_rebote = vx_impacto * tiempo_vuelo_rebote

    num_frames = 200
    tiempo_total = tiempo_bajada + tiempo_vuelo_rebote
    t_values = np.linspace(0, tiempo_total * 1.1, num_frames) # Un poco más para ver el final

    x_trajectory = []
    y_trajectory = []
    frames = []

    # Puntos de la trayectoria estática para referencia del plano y el suelo
    x_plano_end_static = longitud_plano * np.cos(angulo_plano_rad)

    for i, t in enumerate(t_values):
        current_x = 0
        current_y = 0
        
        if t <= tiempo_bajada and tiempo_bajada > 0:
            # Bajada por el plano
            s_plano = 0.5 * a_plano * t**2
            current_x = s_plano * np.cos(angulo_plano_rad)
            current_y = altura_inicial - s_plano * np.sin(angulo_plano_rad)
        else:
            # Rebote parabólico
            t_rebote = t - tiempo_bajada
            if t_rebote >= 0 and tiempo_vuelo_rebote > 0:
                current_x = x_plano_end_static + vx_impacto * t_rebote
                current_y = 0 + vy_rebote * t_rebote - 0.5 * g * t_rebote**2
                # Asegurarse de que el objeto no vaya por debajo del suelo si es una colisión inelástica (e=0)
                if e_impacto == 0:
                    current_y = max(0, current_y)
            else:
                # Si no hay rebote o tiempo fuera de rango, permanece en el punto de impacto en el suelo
                current_x = x_plano_end_static + distancia_horizontal_rebote
                current_y = 0
        
        x_trajectory.append(current_x)
        y_trajectory.append(current_y)

        frame_data = [
            go.Scatter(x=[current_x], y=[current_y],
                       mode='markers', marker=dict(size=m_obj*10 + 10, color='blue', symbol='circle'))
        ]
        frames.append(go.Frame(data=frame_data, name=f'{t:.2f}'))

    fig = go.Figure(
        data=[
            # Traza para la trayectoria completa (estática, de fondo)
            go.Scatter(x=x_trajectory, y=y_trajectory,
                       mode='lines', line=dict(color='orange', width=2, dash='dot'),
                       name='Trayectoria Completa', showlegend=True), # Muestra en leyenda
            # Objeto inicial (posición del primer frame)
            go.Scatter(x=[x_trajectory[0]], y=[y_trajectory[0]],
                       mode='markers', marker=dict(size=m_obj*10 + 10, color='blue', symbol='circle'),
                       name='Objeto Móvil') # Objeto animado (se actualiza en frames)
        ],
        layout=go.Layout(
            xaxis=dict(range=[-0.5, max(x_plano_end_static, x_plano_end_static + distancia_horizontal_rebote) + 1], autorange=False, title="Posición X (m)"),
            yaxis=dict(range=[-0.5, altura_inicial * 1.2], autorange=False, title="Posición Y (m)", # Ajuste para asegurar que el suelo esté visible
                       scaleanchor="x", scaleratio=1),
            title='Animación: Caída y Rebote en Plano Inclinado',
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}])])],
            shapes=[
                # Plano inclinado estático de fondo
                dict(type="line", x0=0, y0=altura_inicial, x1=x_plano_end_static, y1=0,
                     line=dict(color="light green", width=3)),
                # Suelo estático de fondo (se extiende hasta el final de la trayectoria rebotada)
                dict(type="line", x0=0, y0=0, x1=x_plano_end_static + distancia_horizontal_rebote + 1, y1=0,
                     line=dict(color="red", width=3, dash='solid')) # Asegura que sea una línea sólida para el piso
            ],
            plot_bgcolor='rgba(240,240,240,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            showlegend=True # Asegura que la leyenda de la trayectoria se muestre
        ),
        frames=frames
    )
    return fig


# -------------------- Aplicación Principal Streamlit --------------------

st.sidebar.title("Menú de Simulaciones")
simulation_type = st.sidebar.radio(
     "Selecciona una opción:",
     ("Fundamentos Teóricos",
     "Simulación de Colisión 1D",
     "Simulación de Colisión 2D",
     "Cálculo de Impulso y Fuerza Promedio",
     "Péndulo Balístico",
     "Flecha que se Incrusta en un Saco",
     "Caída por Plano Inclinado + Impacto")
     )
st.sidebar.markdown("---")
st.sidebar.info("¡Experimenta con los parámetros para comprender mejor los conceptos físicos!")

# -------------------- Contenido Principal de la Aplicación --------------------

if simulation_type == "Fundamentos Teóricos":
    st.header("📚 Fundamentos Teóricos de Impulso y Cantidad de Movimiento Lineal")
    st.markdown("""
        La **cantidad de movimiento lineal** (o momento lineal) es una propiedad fundamental de los objetos en movimiento.
        Se define como el producto de la **masa** de un objeto y su **velocidad**. Es una **cantidad vectorial**, lo que
        significa que tiene magnitud y dirección.

        ---

        ### Definiciones Clave:

        * Cantidad de Movimiento Lineal ($\\vec{P}$):
           $\\vec{P}$ = m $\\vec{v}$ 
            * $m$ = masa del objeto (kg)
            * $\\vec{v}$ = velocidad del objeto (m/s)
            * Unidades: kg·m/s

        * **Impulso ($\\vec{J}$):**
            Representa el cambio en la cantidad de movimiento de un objeto. También puede verse como
            la fuerza neta aplicada sobre un objeto durante un intervalo de tiempo.
            $$ \\vec{J} = \\Delta \\vec{p} = \\vec{p}_{final} - \\vec{p}_{inicial} $$
            $$ \\vec{J} = \\vec{F}_{promedio} \\Delta t $$
            Donde:
            * $\\Delta \\vec{p}$ = cambio en la cantidad de movimiento
            * $\\vec{F}_{promedio}$ = fuerza promedio neta aplicada (N)
            * $\\Delta t$ = intervalo de tiempo (s)
            * Unidades: N·s (que es equivalente a kg·m/s)

        * **Teorema del Impulso y la Cantidad de Movimiento:**
            Establece que el impulso aplicado a un objeto es igual al cambio en su cantidad de movimiento.
            Este teorema es crucial para analizar colisiones e impactos donde las fuerzas son grandes y actúan por poco tiempo.

        ---

        ### Colisiones y Conservación del Momento:

        En un **sistema aislado** (donde no actúan fuerzas externas netas), la cantidad de movimiento lineal total del sistema
        permanece constante. Esto es conocido como la **Ley de Conservación de la Cantidad de Movimiento Lineal**.

        $$ \\vec{p}_{total, inicial} = \\vec{p}_{total, final} $$
        Esto es particularmente útil para analizar **colisiones**, ya que la cantidad de movimiento total antes de la colisión es igual
        a la cantidad de movimiento total después de la colisión.

        * **Colisiones Elásticas ($e=1$):**
            Tanto la cantidad de movimiento lineal como la **energía cinética total** del sistema se conservan. Los objetos "rebotan" perfectamente.

        * **Colisiones Inelásticas ($e=0$):**
            La cantidad de movimiento lineal se conserva, pero la energía cinética total **no se conserva** (parte de la energía se transforma en calor, sonido, deformación, etc.).
            En una **colisión perfectamente inelástica**, los objetos se pegan y se mueven como uno solo después del impacto.

        * **Coeficiente de Restitución ($e$):**
            Es una medida de la "elasticidad" de una colisión entre dos objetos. Se define como la razón de la velocidad relativa de separación
            a la velocidad relativa de aproximación.
            $$ e = - \\frac{(\\vec{v}_{2,final} - \\vec{v}_{1,final})}{(\\vec{v}_{2,inicial} - \\vec{v}_{1,inicial})} $$
            * $e = 1$ para colisiones perfectamente elásticas.
            * $e = 0$ para colisiones perfectamente inelásticas.
            * $0 < e < 1$ para colisiones inelásticas.

        ---

        ### **Ejemplos de Aplicación:**
        Veremos cómo estos principios se aplican en simulaciones de colisiones, péndulos balísticos, y más.
    """)

elif simulation_type == "Cálculo de Impulso y Fuerza Promedio":
    st.header("⚡ Cálculo de Impulso ($\mathbf{J}$) y Fuerza Promedio ($\mathbf{F}_{promedio}$)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eleccion = st.radio(
            "¿Qué deseas calcular?",
            ("Impulso (dada Fuerza y Tiempo)", "Fuerza Promedio (dado Impulso y Tiempo)")
        )
        
    with col2:
        if eleccion == "Impulso (dada Fuerza y Tiempo)":
            F = st.number_input("Fuerza aplicada $F$ (N)", value=100.0, min_value=0.0, step=10.0)
            dt = st.number_input("Intervalo de tiempo $\\Delta t$ (s)", value=0.1, min_value=0.001, max_value=10.0, step=0.01)
            
            impulso_calc, _ = calcular_impulso_fuerza("impulso", F, dt)
            
            st.markdown("### Resultados")
            st.metric("Impulso $J$", f"{impulso_calc:.2f} N·s")
            
            # --- Digitalización de Datos (Tabla) ---
            datos_j = {
                "Parámetro": ["Fuerza (N)", "Tiempo (s)", "Impulso (N·s)"],
                "Valor": [f"{F:.2f}", f"{dt:.2f}", f"{impulso_calc:.2f}"]
            }
            st.dataframe(datos_j, hide_index=True)
            
        else: # Cálculo de Fuerza Promedio
            J = st.number_input("Impulso $J$ (N·s)", value=15.0, min_value=0.0, step=1.0)
            dt = st.number_input("Intervalo de tiempo $\\Delta t$ (s)", value=0.05, min_value=0.001, max_value=10.0, step=0.005)
            
            fuerza_calc, _ = calcular_impulso_fuerza("fuerza_promedio", J, dt)

            st.markdown("### Resultados")
            st.metric("Fuerza Promedio $F_{promedio}$", f"{fuerza_calc:.2f} N")
            
            # --- Digitalización de Datos (Tabla) ---
            datos_f = {
                "Parámetro": ["Impulso (N·s)", "Tiempo (s)", "Fuerza Promedio (N)"],
                "Valor": [f"{J:.2f}", f"{dt:.2f}", f"{fuerza_calc:.2f}"]
            }
            st.dataframe(datos_f, hide_index=True)

    st.markdown("---")
    st.markdown("El **Teorema de Impulso-Momento** establece que $J = F \\cdot \\Delta t = \\Delta P$. Una pequeña variación de tiempo ($\Delta t$) con un gran Impulso ($J$) implica una **Fuerza Promedio** muy grande. Es por eso que los airbags y las colchonetas son esenciales para alargar $\Delta t$ y reducir $F_{promedio}$.")

### 2. Simulación de Colisión 1D (Unidimensional)
elif simulation_type == "Simulación de Colisión 1D":
    st.header("💥 Colisión Unidimensional: Conservación del Momento")

    # Parámetros de entrada
    col1, col2, col3 = st.columns(3)
    with col1:
        m1 = st.slider("Masa $m_1$ (kg)", 0.5, 5.0, 1.0, 0.1)
        v1_i = st.slider("Velocidad Inicial $v_{1i}$ (m/s)", -5.0, 5.0, 3.0, 0.1)
    with col2:
        m2 = st.slider("Masa $m_2$ (kg)", 0.5, 5.0, 2.0, 0.1)
        v2_i = st.slider("Velocidad Inicial $v_{2i}$ (m/s)", -5.0, 5.0, -1.0, 0.1)
    with col3:
        e = st.slider("Coeficiente de Restitución $e$", 0.0, 1.0, 1.0, 0.01, help="0: Perfectamente Inelástica (se pegan) | 1: Elástica (ideal)")
    
    # Cálculos
    v1_f, v2_f = simular_colision_1d(m1, v1_i, m2, v2_i, e)

    # Cantidades de movimiento
    P_total_i = m1 * v1_i + m2 * v2_i
    P_total_f = m1 * v1_f + m2 * v2_f
    
    # Energía Cinética
    K_i = 0.5 * m1 * v1_i**2 + 0.5 * m2 * v2_i**2
    K_f = 0.5 * m1 * v1_f**2 + 0.5 * m2 * v2_f**2
    perdida_K = K_i - K_f

    st.markdown("---")
    
    # --- Digitalización de Datos (Métricas) ---
    st.subheader("Resultados de la Colisión")
    colA, colB, colC = st.columns(3)
    colA.metric("Velocidad Final $v_{1f}$", f"{v1_f:.2f} m/s", f"{v1_f - v1_i:.2f} $\\Delta v$")
    colB.metric("Velocidad Final $v_{2f}$", f"{v2_f:.2f} m/s", f"{v2_f - v2_i:.2f} $\\Delta v$")
    colC.metric("Pérdida de Energía Cinética", f"{perdida_K:.2f} J")

    # --- Digitalización de Datos (Tabla Resumen) ---
    st.markdown("#### Resumen de Momento y Energía")
    datos_colision = {
        "Parámetro": ["Momento Total Inicial $P_{total, i}$ (kg·m/s)", "Momento Total Final $P_{total, f}$ (kg·m/s)", "Energía Cinética Inicial $K_i$ (J)", "Energía Cinética Final $K_f$ (J)"],
        "Valor": [f"{P_total_i:.2f}", f"{P_total_f:.2f}", f"{K_i:.2f}", f"{K_f:.2f}"]
    }
    st.dataframe(datos_colision, hide_index=True)
    
    st.markdown("---")
    st.subheader("Animación de la Colisión")
    fig_1d = plot_colision_1d_animacion(m1, v1_i, m2, v2_i, e)
    st.plotly_chart(fig_1d, use_container_width=True)

### 3. Simulación de Colisión 2D (Bidimensional)
elif simulation_type == "Simulación de Colisión 2D":
    st.header("🌐 Colisión Bidimensional: Vectores de Momento")

    # Parámetros de entrada
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Objeto 1 (Azul)**")
        m1 = st.slider("$m_1$ (kg)", 0.5, 3.0, 1.0, 0.1, key="m1_2d")
        v1_ix = st.slider("$v_{1ix}$ (m/s)", -5.0, 5.0, 3.0, 0.1, key="v1ix_2d")
    with col2:
        v1_iy = st.slider("$v_{1iy}$ (m/s)", -5.0, 5.0, 0.0, 0.1, key="v1iy_2d")
        v1_i_magnitud = np.sqrt(v1_ix**2 + v1_iy**2)
        st.metric("Magnitud $v_{1i}$", f"{v1_i_magnitud:.2f} m/s")

    with col3:
        st.markdown("**Objeto 2 (Rojo)**")
        m2 = st.slider("$m_2$ (kg)", 0.5, 3.0, 1.5, 0.1, key="m2_2d")
        v2_ix = st.slider("$v_{2ix}$ (m/s)", -5.0, 5.0, -1.0, 0.1, key="v2ix_2d")
    with col4:
        v2_iy = st.slider("$v_{2iy}$ (m/s)", -5.0, 5.0, 1.0, 0.1, key="v2iy_2d")
        v2_i_magnitud = np.sqrt(v2_ix**2 + v2_iy**2)
        st.metric("Magnitud $v_{2i}$", f"{v2_i_magnitud:.2f} m/s")

    st.markdown("---")
    colA, colB = st.columns([1, 3])
    with colA:
        e = st.slider("Coeficiente de Restitución $e$", 0.0, 1.0, 0.8, 0.01, key="e_2d")
        angulo_impacto = st.slider("Ángulo de la Línea de Impacto (grados)", 0, 360, 45, 5, key="angle_2d")
    
    # Cálculos
    (v1fx, v1fy), (v2fx, v2fy) = simular_colision_2d(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto)

    v1_f_magnitud = np.sqrt(v1fx**2 + v1fy**2)
    v2_f_magnitud = np.sqrt(v2fx**2 + v2fy**2)
    
    P_ix = m1 * v1_ix + m2 * v2_ix
    P_iy = m1 * v1_iy + m2 * v2_iy
    P_fx = m1 * v1fx + m2 * v2fx
    P_fy = m1 * v1fy + m2 * v2fy
    
    # --- Digitalización de Datos (Métricas) ---
    st.subheader("Resultados Finales (Componentes)")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Objeto 1 | $v_{1fx}$", f"{v1fx:.2f} m/s")
    colB.metric("Objeto 1 | $v_{1fy}$", f"{v1fy:.2f} m/s")
    colC.metric("Objeto 2 | $v_{2fx}$", f"{v2fx:.2f} m/s")
    colD.metric("Objeto 2 | $v_{2fy}$", f"{v2fy:.2f} m/s")
    
    st.markdown("#### Conservación del Momento (debe ser constante)")
    colE, colF, colG, colH = st.columns(4)
    colE.metric("Momento $X$ Inicial", f"{P_ix:.2f} Ns")
    colF.metric("Momento $X$ Final", f"{P_fx:.2f} Ns")
    colG.metric("Momento $Y$ Inicial", f"{P_iy:.2f} Ns")
    colH.metric("Momento $Y$ Final", f"{P_fy:.2f} Ns")
    
    st.markdown("---")
    st.subheader("Visualización de Trayectorias")
    fig_2d = plot_colision_2d_trayectorias(m1, v1_ix, v1_iy, m2, v2_ix, v2_iy, e, angulo_impacto)
    st.plotly_chart(fig_2d, use_container_width=True)

### 4. Péndulo Balístico
elif simulation_type == "Péndulo Balístico":
    st.header("🎯 Péndulo Balístico: Colisión Inelástica y Conservación de Energía")
    
    col1, col2 = st.columns(2)
    with col1:
        m_bala = st.number_input("Masa de la Bala $m_b$ (kg)", value=0.01, min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
        v_bala_i = st.number_input("Velocidad Inicial de la Bala $v_{bi}$ (m/s)", value=300.0, min_value=10.0, max_value=500.0, step=10.0)
    with col2:
        m_caja = st.number_input("Masa de la Caja $m_c$ (kg)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        largo_pendulo = st.number_input("Largo de la Cuerda $L$ (m)", value=2.0, min_value=0.1, max_value=5.0, step=0.1)
        
    # Cálculos
    v_sistema = calcular_v_sistema_pendulo(m_caja, m_bala, v_bala_i)
    h_max = calcular_h_max_pendulo(m_caja, m_bala, v_bala_i)

    st.markdown("---")
    st.subheader("Resultados de la Colisión y Oscilación")
    
    colA, colB = st.columns(2)
    colA.metric("Velocidad del Sistema (Caja + Bala) Justo Después del Impacto $v_{f}$", f"{v_sistema:.2f} m/s")
    colB.metric("Altura Máxima Alcanzada $h_{max}$", f"{h_max:.2f} m")

    # --- Digitalización de Datos (Tabla Resumen) ---
    st.markdown("#### Detalle de la Conservación de Momento/Energía")
    
    P_i = m_bala * v_bala_i
    P_f = (m_bala + m_caja) * v_sistema
    K_i = 0.5 * m_bala * v_bala_i**2
    K_f_sistema = 0.5 * (m_bala + m_caja) * v_sistema**2
    U_max = (m_bala + m_caja) * g * h_max
    
    datos_pendulo = {
        "Etapa": ["Antes del Impacto", "Después del Impacto", "Punto más Alto"],
        "Momento Lineal (Ns)": [f"{P_i:.2f}", f"{P_f:.2f}", "0 (en Y)"],
        "Energía Cinética (J)": [f"{K_i:.2f}", f"{K_f_sistema:.2f}", "0"],
        "Energía Potencial (J)": ["0", "0 (referencia)", f"{U_max:.2f}"]
    }
    st.dataframe(datos_pendulo, hide_index=True)
    
    st.markdown(f"**Nota:** El momento lineal se conserva estrictamente en la colisión ($P_i \\approx P_f$). La Energía Cinética se pierde significativamente ($K_i$ vs $K_f$) ya que la colisión es **perfectamente inelástica** ($e=0$).")
    
    st.markdown("---")
    st.subheader("Animación de Péndulo Balístico")
    fig_pendulo = plot_pendulo_balistico_animacion(m_bala, m_caja, v_bala_i, largo_pendulo)
    st.plotly_chart(fig_pendulo, use_container_width=True)

### 5. Flecha que se Incrusta en un Saco (Movimiento con Fricción)

```python
elif simulation_type == "Flecha que se Incrusta en un Saco":
    st.header("🏹 Flecha y Saco: Colisión Inelástica y Disipación por Fricción")

    col1, col2 = st.columns(2)
    with col1:
        m_flecha = st.number_input("Masa de la Flecha $m_f$ (kg)", value=0.1, min_value=0.01, max_value=0.5, step=0.01)
        v_flecha_i = st.number_input("Velocidad Inicial de la Flecha $v_{fi}$ (m/s)", value=50.0, min_value=1.0, max_value=100.0, step=1.0)
    with col2:
        m_saco = st.number_input("Masa del Saco $m_s$ (kg)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
        mu_k = st.slider("Coeficiente de Fricción Cinética $\mu_k$", 0.0, 1.0, 0.3, 0.05, help="Coeficiente de fricción entre el saco y la superficie.")
        
    # Cálculos
    v_sistema_i, F_friccion, distancia_detencion = simular_flecha_saco(m_flecha, v_flecha_i, m_saco, mu_k)
    
    st.markdown("---")
    st.subheader("Resultados del Proceso Físico")

    colA, colB, colC = st.columns(3)
    colA.metric("Velocidad Inicial del Sistema $v_{sistema}$", f"{v_sistema_i:.2f} m/s")
    colB.metric("Fuerza de Fricción $F_k$", f"{F_friccion:.2f} N")
    colC.metric("Distancia Recorrida hasta Detenerse $d$", f"{distancia_detencion:.2f} m")

    # --- Digitalización de Datos (Tabla Resumen) ---
    st.markdown("#### Análisis de Momento y Trabajo")
    
    m_total = m_flecha + m_saco
    K_sistema_i = 0.5 * m_total * v_sistema_i**2
    trabajo_friccion = F_friccion * distancia_detencion
    
    datos_saco = {
        "Parámetro": ["Momento Inicial Flecha $P_{fi}$ (Ns)", "Momento Final Sistema $P_{fs}$ (Ns)", "Energía Cinética Inicial Sistema $K_{si}$ (J)", "Trabajo de la Fricción $W_f$ (J)"],
        "Valor": [f"{m_flecha * v_flecha_i:.2f}", f"{m_total * v_sistema_i:.2f}", f"{K_sistema_i:.2f}", f"{-trabajo_friccion:.2f}"]
    }
    st.dataframe(datos_saco, hide_index=True)
    
    st.markdown("El principio de **Trabajo y Energía** indica que la pérdida de Energía Cinética ($K_{si}$) debe ser igual al trabajo realizado por la fuerza de fricción ($W_f$). Nota: $W_f$ es negativo porque actúa en contra del movimiento. La energía inicial se disipa por el rozamiento.")

    st.markdown("---")
    st.subheader("Animación de la Disipación de Movimiento")
    fig_flecha_saco = plot_flecha_saco_animacion(m_flecha, v_flecha_i, m_saco, mu_k)
    st.plotly_chart(fig_flecha_saco, use_container_width=True)

### 6. Caída por Plano Inclinado + Impacto
elif simulation_type == "Caída por Plano Inclinado + Impacto":
    st.header("🎢 Caída con Impacto: Conservación de Momento y Coeficiente de Restitución")

    col1, col2 = st.columns(2)
    with col1:
        m_obj = st.number_input("Masa del Objeto $m$ (kg)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
        altura_i = st.number_input("Altura Inicial $h$ (m)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
        angulo_plano = st.slider("Ángulo del Plano Inclinado $\\theta$ (grados)", 10, 80, 45, 1)
    with col2:
        mu_k_plano = st.slider("Coef. Fricción en el Plano $\mu_k$", 0.0, 1.0, 0.2, 0.05)
        e_impacto = st.slider("Coeficiente de Restitución del Impacto $e$", 0.0, 1.0, 0.7, 0.05)
        
    # Cálculos
    (a_plano, v_final_plano, vx_impacto, vy_impacto,
     vy_rebote, altura_max_rebote, distancia_horizontal_rebote) = simular_caida_plano_impacto(
         m_obj, altura_i, angulo_plano, mu_k_plano, e_impacto
     )

    st.markdown("---")
    st.subheader("Resultados del Proceso (Plano y Rebote)")
    
    colA, colB, colC = st.columns(3)
    colA.metric("Aceleración en el Plano $a$", f"{a_plano:.2f} m/s$^2$")
    colB.metric("Velocidad al Final del Plano $v_{final}$", f"{v_final_plano:.2f} m/s")
    colC.metric("Velocidad Vertical de Rebote $|v_{y,rebote}|$", f"{vy_rebote:.2f} m/s")

    colD, colE = st.columns(2)
    colD.metric("Altura Máxima del Rebote $h_{max}$", f"{altura_max_rebote:.2f} m")
    colE.metric("Distancia Horizontal de Rebote $\\Delta x$", f"{distancia_horizontal_rebote:.2f} m")

    # --- Digitalización de Datos (Tabla Resumen) ---
    st.markdown("#### Análisis de Cantidad de Movimiento e Impulso en el Impacto")
    
    P_y_i = m_obj * vy_impacto # Momento Vertical antes del impacto
    P_y_f = m_obj * vy_rebote # Momento Vertical después del impacto
    J_y = P_y_f - P_y_i # Impulso Vertical en el impacto

    datos_impacto = {
        "Parámetro de Impacto (Vertical)": ["Momento Inicial $P_y^i$ (Ns)", "Momento Final $P_y^f$ (Ns)", "Cambio de Momento $\\Delta P_y$ (Ns)", "Impulso Total $J_y$ (Ns)"],
        "Valor": [f"{P_y_i:.2f}", f"{P_y_f:.2f}", f"{P_y_f - P_y_i:.2f}", f"{J_y:.2f}"]
    }
    st.dataframe(datos_impacto, hide_index=True)

    st.markdown("El **impulso vertical** ($J_y$) representa la fuerza promedio ejercida por el suelo durante el corto tiempo de contacto para cambiar la dirección de la velocidad vertical del objeto.")

    st.markdown("---")
    st.subheader("Animación: Caída, Impacto y Rebote")
    fig_caida_impacto = plot_caida_plano_impacto_animacion(m_obj, altura_i, angulo_plano, mu_k_plano, e_impacto)
    st.plotly_chart(fig_caida_impacto, use_container_width=True)
st.markdown("---")
st.markdown("Desarrollado por Grupo E  para  Física 2.")
