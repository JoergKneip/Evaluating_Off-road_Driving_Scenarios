import cv2

import numpy as np

import sys

import os

import pandas as pd

import plotly.graph_objects as go

import time

import plotly.express as px

from datetime import datetime



# WICHTIG: Diese Imports werden für die Berechnung benötigt

from scipy.spatial.distance import cdist

from scipy.interpolate import splprep, splev



# Versuchen, trimesh für 3D-Modelle zu laden

try:

    import trimesh

    TRIMESH_AVAILABLE = True

except ImportError:

    TRIMESH_AVAILABLE = False

    print("Info: 'trimesh' nicht installiert. Nutze Fallback-Boxen.")



# =============================================================================

# 1. KONFIGURATION

# =============================================================================



# --- NEU: DREHPUNKT FÜR NEIGUNG ---

# 0 = Mitte, 1.4 = Hinterachse (ca.), -2.0 = Vorderachse

TILT_PIVOT_X = 1.4



# Pfade (Bitte anpassen falls nötig)

VIDEO_DIR = r"C:\Users\micha\OneDrive\Desktop\Drohnenvideos\Python und Drohne\Video_drone"

STL_FILENAME = r"C:\Users\micha\OneDrive\Desktop\Drohnenvideos\Python und Drohne\G_Klasse_Aussenhaut.stl"

STL_2_FILENAME = r"C:\Users\micha\OneDrive\Desktop\Drohnenvideos\Python und Drohne\AUF_HUEGEL.stl"



# Dynamische Dateinamen

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

EXCEL_FILENAME = os.path.join(VIDEO_DIR, f"drohnen_messdaten_{timestamp}.xlsx")

HTML_FILENAME = os.path.join(VIDEO_DIR, f"fahrzeug_inspektion_{timestamp}.html")



# --- VISUALISIERUNG: FLÄCHEN + KANTEN ---

MODEL_SETTINGS = {

    'scale': 0.001, 'auto_center': True, 'offset': [0.1, 0, 1],

    'rotation': [0, 0, 0],

   

    # G-Klasse: Flächen unsichtbar, nur dunkelgraue Kanten

    'color':  "#8B8886",    # Farbe egal, da unsichtbar

    'opacity': 0.8,      # Fläche komplett transparent

    'flatshading': False,

   

    # Kanten aktivieren

    'show_edges': True,

    'edge_color': "#000000", # Fast Schwarz (sehr technisch)

    'line_width': 1          # Feine Linien

}



MODEL_2_SETTINGS = {

    'scale': 0.001, 'auto_center': True, 'offset': [1, 0, 0.5],

    'rotation': [0, -30, 0],

   

    # Hügel: Flächen unsichtbar, nur braune Kanten

    'color':  '#8B4513',

    'opacity': 1,

    'flatshading': False,



    # Kanten aktivieren

    'show_edges': True,

    'edge_color': "#000000", # Braun (damit man es vom Auto unterscheiden kann)

    'line_width': 1

}



# Performance & Visualisierung

ANALYSIS_FRAME_STRIDE = 1  

RAY_STRIDE = 3              

RAY_LENGTH = 0.5            



# Frame-Steuerung

SKIP_START_FRAMES = 5      

MAX_FRAMES_TO_ANALYZE = 50  



# Analyse Einstellungen

MIN_MARKERS_REQUIRED = 1    

OUTLIER_THRESHOLD = 0.5    

REAL_MARKER_SIZE = 0.22

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

SMOOTHING_ALPHA = 1



# --- MARKER CONFIG ---

MARKER_CONFIG = {

    0: {'pos': [-2.314, -0.010, 0.703], 'rot': [92, -2, -90]},

    2: {'pos': [-2.174, -0.67, 0.568], 'rot': [96, -1, -58]},

    3: {'pos': [-2.174, 0.67, 0.568], 'rot': [98, -2, -122]},

    4: {'pos': [-0.55, -0.876, 0.666], 'rot': [83, 0, 0]},

    5: {'pos': [-0.55, 0.876, 0.666], 'rot': [95, -0.5, 180]},

    6: {'pos': [0.753, -0.877, 0.661], 'rot': [83, 0, 0]},

    7: {'pos': [0.753, 0.877, 0.661], 'rot': [95, -0.5, 180]},

    9: {'pos': [1.851, -0.805, 1.408], 'rot': [80, 0, 0]},

    10: {'pos': [1.851, 0.805, 1.408], 'rot': [82, -0.5, 180]},

    11: {'pos': [2.211, -0.700, 1.038], 'rot': [90, 1.5, 90]},

    12: {'pos': [2.227, 0.442, 0.808], 'rot': [90, 1.5, 90]},

}



# =============================================================================

# 2. MATHEMATISCHE HILFSFUNKTIONEN

# =============================================================================

def euler_to_rotation_matrix(rx, ry, rz):

    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx



def get_transform_matrix(tvec, rvec, is_rvec_rodrigues=True):

    if is_rvec_rodrigues: R, _ = cv2.Rodrigues(rvec)

    else: R = rvec

    T = np.eye(4)

    T[:3, :3] = R

    T[:3, 3] = tvec.flatten()

    return T



def apply_rotation_to_points(points, R):

    pivot_offset = np.array([TILT_PIVOT_X, 0, 0])

    points_shifted = points - pivot_offset

    points_rotated = (R @ points_shifted.T).T

    return points_rotated + pivot_offset



def get_user_tilt():

    print("\n" + "="*50)

    print("NEIGUNGS-KONFIGURATION")

    print("="*50)

    try:

        print("Lateral (Enter für 0):")

        lat_in = input("-> ")

        angle_x = float(lat_in) if lat_in.strip() else 0.0

        print("Vertikal (Enter für 0):")

        vert_in = input("-> ")

        val_vert = float(vert_in) if vert_in.strip() else 0.0

        angle_y = -val_vert

    except ValueError:

        return np.eye(3)

    return euler_to_rotation_matrix(angle_x, angle_y, 0)



# =============================================================================

# 3. ANALYSE

# =============================================================================

def get_camera_pose_in_vehicle_frame(tvec_cam_marker, rvec_cam_marker, marker_id):

    if marker_id not in MARKER_CONFIG: return None, None, None

    config = MARKER_CONFIG[marker_id]

    H_cm = get_transform_matrix(tvec_cam_marker, rvec_cam_marker)

    H_mc = np.linalg.inv(H_cm)

    R_cfg = euler_to_rotation_matrix(*config['rot'])

    t_cfg = np.array(config['pos'])

    H_mv = get_transform_matrix(t_cfg, R_cfg, is_rvec_rodrigues=False)

    H_vc = H_mv @ H_mc

    return H_vc[:3, 3], H_vc[:3, 2], H_vc[:3, 1]



def get_dji_mini4_calibration(width, height):

    focal_length_px = width * (24.0 / 36.0)

    camera_matrix = np.array([[focal_length_px, 0, width/2.0], [0, focal_length_px, height/2.0], [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    return camera_matrix, dist_coeffs



def run_analysis():

    print("\nSTART: Analyse...")

    if not os.path.exists(VIDEO_DIR):

        print(f"Fehler: Ordner nicht gefunden: {VIDEO_DIR}")

        return False

       

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

    if not video_files:

        print("Keine Videos gefunden.")

        return False

   

    cam_init_done = False

    aruco_dict, params, cam_matrix, dist_coeffs, obj_points = None, None, None, None, None

    all_excel_data = []



    for v_idx, filename in enumerate(video_files):

        video_path = os.path.join(VIDEO_DIR, filename)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened(): continue



        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cam_init_done:

            w, h = int(cap.get(3)), int(cap.get(4))

            cam_matrix, dist_coeffs = get_dji_mini4_calibration(w, h)

            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

            params = cv2.aruco.DetectorParameters()

            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            mh = REAL_MARKER_SIZE / 2.0

            obj_points = np.array([[-mh, mh, 0], [mh, mh, 0], [mh, -mh, 0], [-mh, -mh, 0]], dtype=np.float32)

            cam_init_done = True



        pose_history = {}

        frame_count = 0

        print(f"Video startet: {filename} ({total_frames} Frames)...")



        fps_start_time = time.time()

        frames_since_last_print = 0

        calc_interval = 30



        try:

            while True:

                ret, frame = cap.read()

                if not ret: break

                frame_count += 1

               

                if frame_count <= SKIP_START_FRAMES: continue

               

                if MAX_FRAMES_TO_ANALYZE > 0:

                    if frame_count > (SKIP_START_FRAMES + MAX_FRAMES_TO_ANALYZE):

                        print(f"  -> Limit erreicht. Nächstes Video.")

                        break



                frames_since_last_print += 1

                if frames_since_last_print >= calc_interval:

                    current_time = time.time()

                    elapsed = current_time - fps_start_time

                    if elapsed > 0:

                        fps = frames_since_last_print / elapsed

                        print(f"  Speed: {fps:.1f} FPS (Frame {frame_count}/{total_frames})")

                    fps_start_time = current_time

                    frames_since_last_print = 0

               

                if frame_count % ANALYSIS_FRAME_STRIDE != 0: continue



                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

               

                row_data = {"Source_Video": filename, "Frame": frame_count}

                candidates_pos, candidates_view, candidates_up, candidates_ids = [], [], [], []



                if ids is not None:

                    ids_flat = ids.flatten()

                    for i, mid in enumerate(ids_flat):

                        if mid in MARKER_CONFIG:

                            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                            if success:

                                if mid in pose_history:

                                    last_r, last_t = pose_history[mid]

                                    tvec = SMOOTHING_ALPHA * tvec + (1 - SMOOTHING_ALPHA) * last_t

                                    rvec = SMOOTHING_ALPHA * rvec + (1 - SMOOTHING_ALPHA) * last_r

                                pose_history[mid] = (rvec, tvec)

                               

                                pos, view, up = get_camera_pose_in_vehicle_frame(tvec, rvec, mid)

                                if pos is not None:

                                    candidates_pos.append(pos)

                                    candidates_view.append(view)

                                    candidates_up.append(up)

                                    candidates_ids.append(mid)



                valid_pos, valid_view, valid_up = [], [], []

                if len(candidates_pos) >= MIN_MARKERS_REQUIRED:

                    median_pos = np.median(candidates_pos, axis=0)

                    for i, p in enumerate(candidates_pos):

                        if np.linalg.norm(p - median_pos) < OUTLIER_THRESHOLD:

                            valid_pos.append(p)

                            valid_view.append(candidates_view[i])

                            valid_up.append(candidates_up[i])

                   

                    if len(valid_pos) >= MIN_MARKERS_REQUIRED:

                        mean_pos = np.mean(valid_pos, axis=0)

                        mean_view = np.mean(valid_view, axis=0)

                        mean_up = np.mean(valid_up, axis=0)

                       

                        if np.linalg.norm(mean_view) > 0: mean_view /= np.linalg.norm(mean_view)

                        if np.linalg.norm(mean_up) > 0: mean_up /= np.linalg.norm(mean_up)



                        row_data.update({

                            "AVG_Vehicle_X": mean_pos[0], "AVG_Vehicle_Y": mean_pos[1], "AVG_Vehicle_Z": mean_pos[2],

                            "View_X": mean_view[0], "View_Y": mean_view[1], "View_Z": mean_view[2],

                            "Up_X": mean_up[0], "Up_Y": mean_up[1], "Up_Z": mean_up[2],

                            "Markers_Used": str(candidates_ids)

                        })

                        all_excel_data.append(row_data)



                if cv2.waitKey(1) & 0xFF == ord('q'): break

       

        except KeyboardInterrupt:

            print("\n!!! Unterbrechung durch Benutzer. !!!")

            break

       

        cap.release()



    cv2.destroyAllWindows()

    if all_excel_data:

        pd.DataFrame(all_excel_data).to_excel(EXCEL_FILENAME, index=False)

        print(f"Daten gespeichert: {EXCEL_FILENAME}")

        return True

    return False



# =============================================================================

# 4. VISUALISIERUNG (FLÄCHEN + KANTEN OVERLAY - FIX)

# =============================================================================

def get_tilted_model_traces(filename, settings, tilt_matrix, name_label):

    """

    Lädt STL robust (Scene-Support) und liefert Traces für Fläche & Kanten.

    """

    if not TRIMESH_AVAILABLE or not os.path.exists(filename):

        print(f"Datei nicht gefunden oder trimesh fehlt: {filename}")

        return []

   

    traces = []

    try:

        mesh = trimesh.load(filename)

       

        # FIX 1: Falls es eine Szene ist (mehrere Teile), verschmelzen

        if isinstance(mesh, trimesh.Scene):

            print(f"Info: {name_label} ist eine Szene, verschmelze Geometrie...")

            mesh = mesh.dump(concatenate=True)



        if settings.get('auto_center', False): mesh.vertices -= mesh.centroid

       

        # Transformationen

        rot_local = euler_to_rotation_matrix(*settings['rotation'])

        M_local = np.eye(4)

        M_local[:3, :3] = rot_local

        mesh.apply_transform(M_local)

        mesh.apply_scale(settings['scale'])

        mesh.apply_translation(settings['offset'])



        # Neigung

        mesh.apply_translation([-TILT_PIVOT_X, 0, 0])

        M_tilt = np.eye(4)

        M_tilt[:3, :3] = tilt_matrix

        mesh.apply_transform(M_tilt)

        mesh.apply_translation([TILT_PIVOT_X, 0, 0])

       

        v, f = mesh.vertices, mesh.faces



        # 1. FLÄCHE (Bunt) - Wird immer erstellt

        traces.append(go.Mesh3d(

            x=v.T[0], y=v.T[1], z=v.T[2],

            i=f.T[0], j=f.T[1], k=f.T[2],

            color=settings['color'],

            opacity=settings['opacity'],

            name=name_label,

            flatshading=settings.get('flatshading', False),

            lighting=dict(ambient=0.6, diffuse=0.9, roughness=0.1) # Standard-Licht

        ))



        # 2. KANTEN (Schwarz) - Optional & Fehlertolerant

        if settings.get('show_edges', False):

            try:

                # Wir holen nur Kanten, wenn das Mesh nicht riesig ist, um Absturz zu vermeiden

                if len(mesh.faces) < 75000:

                    lines = mesh.vertices[mesh.edges_unique]

                   

                    x_lines = np.full(3 * len(lines), np.nan)

                    y_lines = np.full(3 * len(lines), np.nan)

                    z_lines = np.full(3 * len(lines), np.nan)

                   

                    x_lines[0::3] = lines[:, 0, 0]

                    y_lines[0::3] = lines[:, 0, 1]

                    z_lines[0::3] = lines[:, 0, 2]

                   

                    x_lines[1::3] = lines[:, 1, 0]

                    y_lines[1::3] = lines[:, 1, 1]

                    z_lines[1::3] = lines[:, 1, 2]

                   

                    traces.append(go.Scatter3d(

                        x=x_lines, y=y_lines, z=z_lines,

                        mode='lines',

                        line=dict(color=settings.get('edge_color', 'black'), width=settings.get('line_width', 1)),

                        name=f"{name_label} (Kanten)",

                        showlegend=False

                    ))

                else:

                    print(f"Info: {name_label} hat zu viele Kanten ({len(mesh.faces)} Faces). Kantenmodus deaktiviert für Performance.")

            except Exception as e:

                print(f"Warnung: Kanten konnten für {name_label} nicht berechnet werden ({e}), zeige nur Fläche.")



    except Exception as e:

        print(f"CRITICAL ERROR beim Laden von {filename}: {e}")

        return []

   

    return traces



def get_tilted_box(tilt_matrix):

    l, w, h = 4.8/2, 1.9/2, 1.9  

    x = [-l, l, l, -l, -l, l, l, -l]

    y = [w, w, -w, -w, w, w, -w, -w]

    z = [0, 0, 0, 0, h, h, h, h]

    points = np.column_stack((x, y, z))

    tilted = apply_rotation_to_points(points, tilt_matrix)

    return [go.Mesh3d(x=tilted[:,0], y=tilted[:,1], z=tilted[:,2], alphahull=0, color='gray', opacity=0.3, name='Fallback Box')]



def add_detailed_marker_with_axes(fig, tilt_matrix):

    s = REAL_MARKER_SIZE / 2.0

    axis_len = 0.4

    v_local = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])

    axes_def = [(np.array([axis_len, 0, 0]), 'red', 'X'), (np.array([0, axis_len, 0]), 'green', 'Y'), (np.array([0, 0, axis_len]), 'blue', 'Z')]

    for mid, cfg in MARKER_CONFIG.items():

        if np.allclose(cfg['pos'], [0,0,0]): pass

        R_marker = euler_to_rotation_matrix(*cfg['rot'])

        v_rot = (R_marker @ v_local.T).T

        v_final = apply_rotation_to_points(v_rot + np.array(cfg['pos']), tilt_matrix)

        # Marker Fläche

        fig.add_trace(go.Mesh3d(x=v_final[:,0], y=v_final[:,1], z=v_final[:,2], i=[0, 0], j=[1, 2], k=[2, 3], color='black', opacity=1.0, name=f'M{mid}', showlegend=False))

        center_w = np.mean(v_final, axis=0)

        # Text

        fig.add_trace(go.Scatter3d(x=[center_w[0]], y=[center_w[1]], z=[center_w[2]], mode='text', text=[str(mid)], textfont=dict(color='white', size=14), showlegend=False))

        # Achsen

        for vec_local, color, name in axes_def:

            vec_world = tilt_matrix @ (R_marker @ vec_local)

            end = center_w + vec_world

            fig.add_trace(go.Scatter3d(x=[center_w[0], end[0]], y=[center_w[1], end[1]], z=[center_w[2], end[2]], mode='lines', line=dict(color=color, width=5), name=f'M{mid} {name}', showlegend=False))



def run_visualization(tilt_matrix):

    print("\nSTART: 3D Visualisierung & Kurvenanalyse...")

    if not os.path.exists(EXCEL_FILENAME):

        print(f"Fehler: Excel Datei nicht gefunden: {EXCEL_FILENAME}")

        return

   

    # Daten laden

    try:

        df = pd.read_excel(EXCEL_FILENAME).dropna(subset=["AVG_Vehicle_X"])

    except Exception as e:

        print(f"Fehler beim Laden der Excel: {e}")

        return



    if len(df) == 0:

        print("Excel Tabelle ist leer.")

        return



    if 'Source_Video' not in df.columns:

        df['Source_Video'] = 'Video 1'



    fig = go.Figure()

    video_groups = df.groupby('Source_Video')

    colors = px.colors.qualitative.Set1

    color_idx = 0

    all_centers = []

    all_views = []  



    FILTER_TOLERANCE_METERS = 0.50



    for video_name, group in video_groups:

        pos_local = group[["AVG_Vehicle_X", "AVG_Vehicle_Y", "AVG_Vehicle_Z"]].to_numpy()

        view_local = group[["View_X", "View_Y", "View_Z"]].to_numpy()

       

        pos_world = apply_rotation_to_points(pos_local, tilt_matrix)

        view_world = apply_rotation_to_points(view_local, tilt_matrix)

        current_color = colors[color_idx % len(colors)]

       

        pos_final = pos_world

        view_final = view_world

       

        # Spline-Filterung

        if len(pos_world) > 8:

            try:

                tck, u = splprep(pos_world.T, s=len(pos_world)*0.15, per=False)

                smooth_pos = np.array(splev(u, tck)).T

                distances = np.linalg.norm(pos_world - smooth_pos, axis=1)

                valid_mask = distances <= FILTER_TOLERANCE_METERS

                pos_final = pos_world[valid_mask]

                view_final = view_world[valid_mask]

            except Exception: pass



        if len(pos_final) > 0:

            center_point = np.mean(pos_final, axis=0)

            center_view = np.mean(view_final, axis=0)

            if np.linalg.norm(center_view) > 0:

                center_view = center_view / np.linalg.norm(center_view)

           

            all_centers.append(center_point)

            all_views.append(center_view)



            # Punkt

            fig.add_trace(go.Scatter3d(

                x=[center_point[0]], y=[center_point[1]], z=[center_point[2]],

                mode='markers', marker=dict(size=6, color=current_color, symbol='circle', opacity=1.0),

                name=f'{video_name}', legendgroup=f'{video_name}', showlegend=True

            ))

           

            # Strahl

            end_point = center_point + (center_view * RAY_LENGTH)

            fig.add_trace(go.Scatter3d(

                x=[center_point[0], end_point[0]], y=[center_point[1], end_point[1]], z=[center_point[2], end_point[2]],

                mode='lines', line=dict(color=current_color, width=4),

                name=f'{video_name}', legendgroup=f'{video_name}', showlegend=False, hoverinfo='skip'

            ))

       

        color_idx += 1



    # --- DICHTE BERECHNUNG (NUR RECHTE SEITE: Y > 0) ---

    if len(all_centers) >= 1:

        points_array = np.array(all_centers)

        views_array = np.array(all_views)



        # Filter: Nur Punkte mit Y > 0

        mask_right = points_array[:, 1] > 0

       

        cluster_points = points_array[mask_right]

        cluster_views = views_array[mask_right]

       

        num_valid = len(cluster_points)



        if num_valid > 0:

            print(f"-> Berechne Durchschnitt aus {num_valid} Videos (NUR RECHTS / Y > 0).")



            # Simpler Durchschnitt aller Positionen (RECHTS)

            best_pos = np.mean(cluster_points, axis=0)

            best_view = np.mean(cluster_views, axis=0)

           

            # Vektor normalisieren

            if np.linalg.norm(best_view) > 0:

                best_view = best_view / np.linalg.norm(best_view)



            # Wolke (zeigt den relevanten Bereich)

            fig.add_trace(go.Mesh3d(

                x=cluster_points[:,0], y=cluster_points[:,1], z=cluster_points[:,2],

                alphahull=5, color='gold', opacity=0.2, name='Bereich (Rechts)'

            ))



            # Goldener Diamant (Bester Punkt)

            fig.add_trace(go.Scatter3d(

                x=[best_pos[0]], y=[best_pos[1]], z=[best_pos[2]],

                mode='markers+text',

                marker=dict(size=15, color='gold', symbol='diamond', line=dict(width=2, color='black')),

                text=["DURCHSCHNITT (RECHTS)"], textposition="top center",

                textfont=dict(size=14, color="black"), name='Schnittmenge (Rechts)'

            ))



            # Goldener Strahl

            end_best = best_pos + (best_view * (RAY_LENGTH * 1.5))

            fig.add_trace(go.Scatter3d(

                x=[best_pos[0], end_best[0]], y=[best_pos[1], end_best[1]], z=[best_pos[2], end_best[2]],

                mode='lines', line=dict(color='gold', width=8), name='Optimale Blickrichtung'

            ))

        else:

            print("-> WARNUNG: Keine Videos auf der rechten Seite (Y > 0) gefunden!")



    # --- MODELLE (JETZT ALS TRACE-LISTE) ---

    car_traces = get_tilted_model_traces(STL_FILENAME, MODEL_SETTINGS, tilt_matrix, 'G-Klasse')

    if car_traces:

        for trace in car_traces: fig.add_trace(trace)

    else:

        # Fallback Box falls gar nichts geladen wurde

        for trace in get_tilted_box(tilt_matrix): fig.add_trace(trace)



    no_tilt_matrix = np.eye(3)

    sensor_traces = get_tilted_model_traces(STL_2_FILENAME, MODEL_2_SETTINGS, no_tilt_matrix, 'Hügel')

    if sensor_traces:

         for trace in sensor_traces: fig.add_trace(trace)

   

    add_detailed_marker_with_axes(fig, tilt_matrix)

   

    fig.update_layout(

        title=f"Analyse: Durchschnitt (Nur Rechts, Y > 0) - Solid + Edges",

        scene=dict(

            xaxis=dict(title='X'),

            yaxis=dict(title='Y'),

            zaxis=dict(title='Z'),

            aspectmode='data'

        )

    )

   

    try:

        fig.write_html(HTML_FILENAME)

        print(f"-> Fertig! Datei erstellt: {HTML_FILENAME}")

        os.startfile(HTML_FILENAME)

    except Exception as e:

        print(f"Fehler beim Speichern/Öffnen: {e}")



if __name__ == "__main__":

    print("Programm gestartet...")

    tilt_matrix = get_user_tilt()

   

    # Analyse starten

    run_analysis()

   

    # Kurz warten

    time.sleep(1)

   

    # Visualisierung starten

    run_visualization(tilt_matrix)