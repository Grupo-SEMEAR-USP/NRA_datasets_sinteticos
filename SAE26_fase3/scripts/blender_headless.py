import bpy
import os
import math
from bpy_extras.object_utils import world_to_camera_view
import random

#Altere o ID para bater com o que outra pessoa gerar (caso dividir carga computacional)

START_ID = 0
END_ID = 20

RESOLUCAO_X = 640
RESOLUCAO_Y = 480

#Copie o caminho da pasta para o dataset em seu computador para os campos abaixo

NOME_PROJETO = "manometros_dataset"
OUTPUT_DIR_IMG = r"D:\dataset_generator\dataset\images"
OUTPUT_DIR_TXT = r"D:\dataset_generator\dataset\labels"
INPUT_DIR_TEXT = r"D:\dataset_generator\floor_texture"

scene = bpy.context.scene
scene.render.resolution_x = RESOLUCAO_X
scene.render.resolution_y = RESOLUCAO_Y
scene.render.image_settings.file_format = 'PNG'

scene.render.engine = 'BLENDER_EEVEE'

scene.eevee.taa_render_samples = 2

scene.eevee.shadow_ray_count = 1

scene.eevee.volumetric_tile_size = '16'
scene.eevee.volumetric_samples = 1

cam_obj = bpy.data.objects.get('camera')
agulha_obj = bpy.data.objects.get('agulha')
man_obj = bpy.data.objects.get('mestre_manometro')
light_obj_1 = bpy.data.objects.get('luz_1')
light_obj_2 = bpy.data.objects.get('luz_2')
piso_obj = bpy.data.objects.get('piso')

lista_cantos = ['bot_l_empty',
                'bot_r_empty',
                'top_r_empty',
                'top_l_empty']

def position_camera(camera_obj):
    
    cz = random.uniform(0.5, 1.5)
    
    phi = math.radians(random.uniform(0, 35))
    theta = random.uniform(0, 2 * math.pi)
    r_xy = cz * math.tan(phi)
    
    cx = max(-1.5, min(1.5, (r_xy * math.cos(theta))))
    cy = max(-1.5, min(1.5, (r_xy * math.sin(theta))))
    
    camera_obj.location = (cx, cy, cz)
    
    return None

def is_object_in_frame(scenery, camera_obj, target_obj):

    point = target_obj.location
    
    coords = world_to_camera_view(scenery, camera_obj, point)
    
    in_view = 0.0 <= coords.x <= 1.0 and 0.0 <= coords.y <= 1.0 and coords.z > 0
    
    return in_view

def position_manometer(manometer_obj, camera_obj, scenery):
    
    tentativa = 0
    
    while tentativa < 100:
        
        manometer_obj.location.x = random.uniform(-0.8, 0.8)
        manometer_obj.location.y = random.uniform(-0.8, 0.8)
        manometer_obj.location.z = 0.001
        
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        
        if is_object_in_frame(scenery, camera_obj, manometer_obj):
            break
        
        tentativa +=1
        
def assure_camera_direction(manometer_obj, camera_obj):
    
    direcao = camera_obj.location - manometer_obj.location
    
    angulo_camera = math.atan2(direcao.y, direcao.x)
    
    angulo_final = random.gauss(angulo_camera, 0.1)
    
    manometer_obj.rotation_euler[2] = angulo_final + (0.55 * math.pi)
    
    return None

def position_lights(light_1, light_2):
    
    x_l1 = random.uniform(-1, 1)
    y_l1 = random.uniform(-1, 1)
    x_l2 = random.uniform(-1, 1)
    y_l2 = random.uniform(-1, 1)
    
    light_1.location = (x_l1, y_l1, 3)
    light_2.location = (x_l2, y_l2, 3)
    
    return None

def randomize_floor_texture(floor_obj, directory):
    
    nodes = floor_obj.active_material.node_tree.nodes
    tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)

    if tex_node:
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_path = os.path.join(directory, random.choice(files))
    
        new_img = bpy.data.images.load(img_path)
        tex_node.image = new_img
        
    return None

def position_needle (needle_obj):

    angulo_grau = random.uniform(0, 300)
    angulo_radianos = math.radians(angulo_grau)
    
    needle_obj.rotation_euler[2] = angulo_radianos
    
    return None

def get_bbox(empties_name):

    coords_x = []
    coords_y = []

    for name in empties_name:

        obj = bpy.data.objects.get(name)

        if obj:

            co_2d = world_to_camera_view(bpy.context.scene,
                                         bpy.data.objects['camera'], 
                                         obj.matrix_world.translation)            
            
            nx = max(0.0, min(1.0, co_2d.x))
            ny = max(0.0, min(1.0, co_2d.y))
            
            coords_x.append(nx)
            coords_y.append(ny)            

    min_x, max_x = min(coords_x), max(coords_x)
    min_y, max_y = min(coords_y), max(coords_y)

    x_center = (min_x + max_x) / 2
    y_center = 1- ((min_y + max_y) / 2)

    
    width = max_x - min_x
    height = max_y - min_y 
    
    if width == 0 or height == 0:
        
        return -1, -1, -1, -1
    
    else:
    
        return x_center, y_center, width, height
    
    
def get_keypoint(empty_name):

        obj = bpy.data.objects.get(empty_name)

        if obj:

            co_2d = world_to_camera_view(bpy.context.scene,
                                         bpy.data.objects['camera'], 
                                         obj.matrix_world.translation)
                                                                       
        obj_x = co_2d.x
        obj_y = 1 - co_2d.y
        
        if obj_x < 0 or obj_y < 0:
            
            v = 1
            
        elif obj_x > 1 or obj_y > 1:
            
            v = 1
            
        else:
            
            v = 2         
    
        return obj_x, obj_y, v
    
def generate_txt(bbox, ept_1, ept_2, ept_3, filepath_txt):

    bbox_x, bbox_y, bbox_w, bbox_h = get_bbox(bbox)
    
    o1_x, o1_y, o1_v = get_keypoint(ept_1)
    o2_x, o2_y, o2_v = get_keypoint(ept_2)
    o3_x, o3_y, o3_v = get_keypoint(ept_3)
    
    linha = f"0 {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f} {o1_x:.6f} {o1_y:.6f} {o1_v} {o2_x:.6f} {o2_y:.6f} {o2_v} {o3_x:.6f} {o3_y:.6f} {o3_v}"
    
    with open(filepath_txt, 'w') as f:
        
        if bbox_x == -1:
            
            pass
            
        else:
            f.write(linha)

def set_obj_visibility(obj, visible):

    obj.hide_render = not visible
    obj.hide_viewport = not visible

    for child in obj.children:

        set_obj_visibility(child, visible)

for i in range (START_ID, END_ID + 1):
    
    file_id = f"{i:04d}"
                
    filepath_img = os.path.join(OUTPUT_DIR_IMG, f"{file_id}.png")
    filepath_txt = os.path.join(OUTPUT_DIR_TXT, f"{file_id}.txt")

    esta_presente = random.random() > 0.20

    set_obj_visibility(man_obj, esta_presente)

    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    if esta_presente:

        position_camera(cam_obj)
        
        position_manometer(man_obj, cam_obj, scene)
        
        assure_camera_direction(man_obj, cam_obj)
        
        position_lights(light_obj_1, light_obj_2)
        
        position_needle(agulha_obj)
        
        randomize_floor_texture(piso_obj, INPUT_DIR_TEXT)

        dg.update()
        scene.render.filepath = filepath_img
        bpy.ops.render.render(write_still=True)

        generate_txt(lista_cantos, 'empty_ponta', 'empty_centro', 'empty_base', filepath_txt)

    else:

        position_camera(cam_obj)
        position_lights(light_obj_1, light_obj_2)
        randomize_floor_texture(piso_obj, INPUT_DIR_TEXT)
        
        dg.update()
        
        scene.render.filepath = filepath_img
        bpy.ops.render.render(write_still=True)

        with open(filepath_txt, 'w') as f:
            f.write("")