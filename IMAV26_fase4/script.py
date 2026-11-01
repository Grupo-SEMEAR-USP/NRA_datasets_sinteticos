import bpy
import os
import math
import random
from bpy_extras.object_utils import world_to_camera_view
import mathutils

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

START_ID = 0
END_ID = 1000

RESOLUCAO_X = 640
RESOLUCAO_Y = 480

NOME_PROJETO = "bombeiros_dataset"
OUTPUT_DIR_IMG = r"/home/joao/dataset_fase4_imav_26/val/images"
OUTPUT_DIR_TXT = r"/home/joao/dataset_fase4_imav_26/val/labels"
INPUT_DIR_TEXT = r"/home/joao/dataset_fase4_imav_26/floor"  # opcional, só é usado se existir um objeto 'piso'

# --- Proporções de cenário no lote inteiro (devem somar 1.0) ---
PROB_NENHUM = 0.20   # nenhum bombeiro visível
PROB_UM = 0.70       # apenas b1 OU b2 visível

# --- IDs de classe no formato YOLO detection ---
CLASS_ID_B1 = 0
CLASS_ID_B2 = 0

# --- Ranges de deslocamento em X/Y (em metros) — ajuste conforme a escala da sua cena ---
PERSON_XY_RANGE = 1.0
BUSH_XY_RANGE = 1.8
LIGHT_XY_RANGE = 1.5

# --- Distância mínima em X/Y entre mestre_b1 e mestre_b2 quando os dois estão visíveis ---
MIN_PERSON_DISTANCE = 1.5

# --- Câmera: distância (raio) e faixa de elevação ---
CAM_DIST_MIN = 3.5
CAM_DIST_MAX = 5.0
CAM_ELEV_MIN_DEG = 15   # 0° = câmera na "horizontal" do alvo, 90° = câmera no zênite
CAM_ELEV_MAX_DEG = 75

# --- Densidade de arbustos ---
BUSH_MIN_VISIBLE_FRAC = 0.6  # fração mínima de arbustos visíveis (aumenta a densidade no frame)

# ============================================================
# SETUP DA CENA / RENDER
# ============================================================

os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
os.makedirs(OUTPUT_DIR_TXT, exist_ok=True)

scene = bpy.context.scene
scene.render.resolution_x = RESOLUCAO_X
scene.render.resolution_y = RESOLUCAO_Y
scene.render.image_settings.file_format = 'PNG'

scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 2
scene.eevee.shadow_ray_count = 1
scene.eevee.volumetric_tile_size = '16'
scene.eevee.volumetric_samples = 1

# ============================================================
# OBJETOS DA CENA
# ============================================================

def exigir_objeto(nome):
    """Busca um objeto pelo nome"""
    obj = bpy.data.objects.get(nome)
    if obj is None:
        raise RuntimeError(
            f"Objeto '{nome}' não encontrado na cena. Confira o nome exato no Outliner "
            f"(maiúsculas/minúsculas e espaços importam)."
        )
    return obj


def garantir_visivel_em_colecoes(obj):
    """Garante que nenhuma Coleção em que o objeto está linkado tenha 'Disable in Renders'
    ativado — isso faria o objeto sumir do render mesmo com hide_render=False nele."""
    for colecao in obj.users_collection:
        colecao.hide_render = False


cam_obj = exigir_objeto('Camera')
mestre_b1_obj = exigir_objeto('mestre_b1')
mestre_b2_obj = exigir_objeto('mestre_b2')

for _mestre in (mestre_b1_obj, mestre_b2_obj):
    garantir_visivel_em_colecoes(_mestre)
    for _filho in _mestre.children:
        garantir_visivel_em_colecoes(_filho)


def coletar_arbustos(prefixo='arb'):
    """Coleta dinamicamente todos os objetos cujo nome comece com o prefixo, para que
    novos arbustos adicionados depois no .blend sejam usados automaticamente (mais densidade)."""
    return [obj for obj in bpy.data.objects if obj.name.startswith(prefixo)]


arbustos_objs = coletar_arbustos('arb')

LUZES_NOMES = ['luz1', 'luz2', 'luz2.001']
luzes_objs = [bpy.data.objects.get(n) for n in LUZES_NOMES if bpy.data.objects.get(n) is not None]

piso_obj = bpy.data.objects.get('piso')  # opcional: se não existir nesta cena, é simplesmente ignorado

MESH_B1 = 'bombeiro_costas'
MESH_B2 = 'bombeiro_frente'

# ============================================================
# FUNÇÕES
# ============================================================

def gerar_lista_cenarios(n):
    """Gera a lista de cenários (nenhum / um_b1 / um_b2 / ambos) pra um lote de n imagens,
    respeitando exatamente as proporções PROB_NENHUM / PROB_UM / (resto), embaralhada.
    Isso GARANTE a proporção no lote inteiro, sem depender de sorteio independente por imagem
    (que em lotes pequenos pode desviar bastante do esperado por pura variância)."""

    n_nenhum = round(PROB_NENHUM * n)
    n_ambos = round((1 - PROB_NENHUM - PROB_UM) * n)
    n_um = n - n_nenhum - n_ambos

    n_um_b1 = n_um // 2
    n_um_b2 = n_um - n_um_b1

    cenarios = (['nenhum'] * n_nenhum) + (['um_b1'] * n_um_b1) + (['um_b2'] * n_um_b2) + (['ambos'] * n_ambos)

    # Corrige eventuais folgas de arredondamento pra garantir exatamente n itens
    while len(cenarios) < n:
        cenarios.append('um_b1')
    while len(cenarios) > n:
        cenarios.pop()

    random.shuffle(cenarios)
    return cenarios


def position_camera(camera_obj):
    dist = random.uniform(CAM_DIST_MIN, CAM_DIST_MAX)
    elev = math.radians(random.uniform(CAM_ELEV_MIN_DEG, CAM_ELEV_MAX_DEG))
    azim = random.uniform(0, 2 * math.pi)

    cz = dist * math.sin(elev)
    r_xy = dist * math.cos(elev)
    cx = r_xy * math.cos(azim)
    cy = r_xy * math.sin(azim)

    camera_obj.location = (cx, cy, cz)

    # Faz a câmera olhar para o centro (0, 0, 0)
    direction = mathutils.Vector((0, 0, 0)) - camera_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()

    return None


def is_object_in_frame(scenery, camera_obj, target_obj):

    point = target_obj.location

    coords = world_to_camera_view(scenery, camera_obj, point)

    in_view = 0.0 <= coords.x <= 1.0 and 0.0 <= coords.y <= 1.0 and coords.z > 0

    return in_view


def position_person(person_obj, camera_obj, scenery, avoid_objs=None,
                     xy_range=PERSON_XY_RANGE, min_distance=MIN_PERSON_DISTANCE,
                     max_tentativas=200):
    """Aleatoriza rotação em Z e posição em X/Y de um boneco (mestre_b1 ou mestre_b2).
    Garante que fique dentro do campo de visão da câmera e, se avoid_objs for passado,
    mantém distância mínima dos outros bonecos já posicionados (evita sobreposição).
    A posição em Z nunca é tocada. Se não achar uma posição válida em max_tentativas,
    cai num fallback perto do ponto que a câmera está mirando, garantindo que o boneco
    apareça na imagem."""

    avoid_objs = avoid_objs or []

    person_obj.rotation_euler[2] = random.uniform(0, 2 * math.pi)

    tentativa = 0
    sucesso = False

    while tentativa < max_tentativas:

        person_obj.location.x = random.uniform(-xy_range, xy_range)
        person_obj.location.y = random.uniform(-xy_range, xy_range)

        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        longe_o_suficiente = all(
            math.hypot(person_obj.location.x - outro.location.x,
                       person_obj.location.y - outro.location.y) >= min_distance
            for outro in avoid_objs
        )

        if longe_o_suficiente and is_object_in_frame(scenery, camera_obj, person_obj):
            sucesso = True
            break

        tentativa += 1

    if not sucesso:
        # Fallback: garante que o boneco apareça na imagem mesmo se não achamos uma posição
        # "ideal". Assume que a câmera está mirando perto da origem (0, 0).
        if avoid_objs:
            base = avoid_objs[0].location
            angulo = random.uniform(0, 2 * math.pi)
            person_obj.location.x = base.x + math.cos(angulo) * (min_distance + 0.2)
            person_obj.location.y = base.y + math.sin(angulo) * (min_distance + 0.2)
        else:
            angulo = random.uniform(0, 2 * math.pi)
            raio = random.uniform(0, 0.3)
            person_obj.location.x = raio * math.cos(angulo)
            person_obj.location.y = raio * math.sin(angulo)

        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

    return sucesso


def randomize_bushes(bush_objs, xy_range=BUSH_XY_RANGE):
    """Sorteia uma quantidade aleatória de arbustos visíveis (com um mínimo definido por
    BUSH_MIN_VISIBLE_FRAC, pra manter mais densidade no frame) e randomiza a posição X/Y
    apenas dos que ficarem visíveis. Z nunca é tocado."""

    if not bush_objs:
        return None

    minimo = max(1, int(len(bush_objs) * BUSH_MIN_VISIBLE_FRAC))
    minimo = min(minimo, len(bush_objs))
    n_visiveis = random.randint(minimo, len(bush_objs))
    visiveis = set(random.sample(bush_objs, n_visiveis))

    for obj in bush_objs:
        visivel = obj in visiveis
        set_obj_visibility(obj, visivel)

        if visivel:
            obj.location.x = random.uniform(-xy_range, xy_range)
            obj.location.y = random.uniform(-xy_range, xy_range)

    return None


def position_lights(luzes, xy_range=LIGHT_XY_RANGE):
    """Randomiza X/Y das luzes, preservando a altura (Z) original de cada uma."""

    for luz in luzes:
        luz.location.x = random.uniform(-xy_range, xy_range)
        luz.location.y = random.uniform(-xy_range, xy_range)

    return None


def randomize_floor_texture(floor_obj, directory):
    nodes = floor_obj.active_material.node_tree.nodes
    tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)

    if tex_node:
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            img_name = random.choice(files)
            img_path = os.path.join(directory, img_name)
            
            # Verifica se a imagem já está no cache do Blender para evitar memory leak
            if img_name in bpy.data.images:
                new_img = bpy.data.images[img_name]
            else:
                new_img = bpy.data.images.load(img_path)
                
            tex_node.image = new_img

    return None


def set_obj_visibility(obj, visible):

    obj.hide_render = not visible
    obj.hide_viewport = not visible

    for child in obj.children:
        set_obj_visibility(child, visible)


def get_bbox(mesh_name, camera_obj):
    scene_ctx = bpy.context.scene
    obj = bpy.data.objects.get(mesh_name)
    
    if not obj or obj.type != 'MESH':
        return -1, -1, -1, -1

    # Atualiza o grafo de dependências para pegar poses/modificadores aplicados
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    
    # Cria uma cópia da malha final para leitura dos vértices
    try:
        mesh = eval_obj.to_mesh()
    except RuntimeError:
        return -1, -1, -1, -1

    coords_x = []
    coords_y = []
    matrix_world = eval_obj.matrix_world

    # Passa por TODOS os vértices do personagem (garante a caixa mais justa possível)
    for v in mesh.vertices:
        world_coord = matrix_world @ v.co
        co_2d = world_to_camera_view(scene_ctx, camera_obj, world_coord)
        
        # Só considera vértices que estão na frente da câmera
        if co_2d.z > 0:
            coords_x.append(co_2d.x)
            coords_y.append(co_2d.y)

    # Limpa a memória para evitar memory leak no Blender
    eval_obj.to_mesh_clear()

    if not coords_x or not coords_y:
        return -1, -1, -1, -1

    # Pega os vértices mais extremos e limita às bordas da imagem (0 a 1)
    min_x = max(0.0, min(coords_x))
    max_x = min(1.0, max(coords_x))
    min_y = max(0.0, min(coords_y))
    max_y = min(1.0, max(coords_y))

    width = max_x - min_x
    height = max_y - min_y

    # Se zerou, o objeto está totalmente fora do frame
    if width <= 0 or height <= 0:
        return -1, -1, -1, -1

    # Calcula centro (Y é invertido na YOLO)
    x_center = (min_x + max_x) / 2
    y_center = 1.0 - ((min_y + max_y) / 2)

    return x_center, y_center, width, height


def generate_txt(pessoas, camera_obj, filepath_txt):
    linhas = []

    # 'pessoas' agora é uma lista de tuplas ex: [(0, 'bombeiro_costas'), ...]
    for classe_id, mesh_name in pessoas:
        bbox_x, bbox_y, bbox_w, bbox_h = get_bbox(mesh_name, camera_obj)
        
        if bbox_x == -1:
            continue
            
        linha = f"{classe_id} {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        linhas.append(linha)

    with open(filepath_txt, 'w') as f:
        f.write("\n".join(linhas))

    return len(linhas)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

TOTAL_IMAGENS = END_ID - START_ID + 1
lista_cenarios = gerar_lista_cenarios(TOTAL_IMAGENS)

print(f"[dataset] Distribuição do lote: "
      f"nenhum={lista_cenarios.count('nenhum')} "
      f"um_b1={lista_cenarios.count('um_b1')} "
      f"um_b2={lista_cenarios.count('um_b2')} "
      f"ambos={lista_cenarios.count('ambos')}")

for idx, i in enumerate(range(START_ID, END_ID + 1)):

    file_id = f"{i:04d}"

    filepath_img = os.path.join(OUTPUT_DIR_IMG, f"{file_id}.png")
    filepath_txt = os.path.join(OUTPUT_DIR_TXT, f"{file_id}.txt")

    cenario = lista_cenarios[idx]
    b1_visivel = cenario in ('um_b1', 'ambos')
    b2_visivel = cenario in ('um_b2', 'ambos')

    set_obj_visibility(mestre_b1_obj, b1_visivel)
    set_obj_visibility(mestre_b2_obj, b2_visivel)

    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    print(f"[{file_id}] cenario={cenario} "
          f"b1_visivel={b1_visivel} (hide_render={mestre_b1_obj.hide_render}) "
          f"b2_visivel={b2_visivel} (hide_render={mestre_b2_obj.hide_render})")

    # --- randomizações da cena ---
    position_camera(cam_obj)
    randomize_bushes(arbustos_objs)
    position_lights(luzes_objs)

    if piso_obj is not None:
        randomize_floor_texture(piso_obj, INPUT_DIR_TEXT)

    if b1_visivel:
        position_person(mestre_b1_obj, cam_obj, scene)
        print(f"    b1 pos=({mestre_b1_obj.location.x:.2f}, {mestre_b1_obj.location.y:.2f})")

    if b2_visivel:
        avoid = [mestre_b1_obj] if b1_visivel else []
        position_person(mestre_b2_obj, cam_obj, scene, avoid_objs=avoid)
        print(f"    b2 pos=({mestre_b2_obj.location.x:.2f}, {mestre_b2_obj.location.y:.2f})")

    dg.update()

    scene.render.filepath = filepath_img
    bpy.ops.render.render(write_still=True)

    pessoas_para_bbox = []
    if b1_visivel:
        pessoas_para_bbox.append((CLASS_ID_B1, MESH_B1))
    if b2_visivel:
        pessoas_para_bbox.append((CLASS_ID_B2, MESH_B2))

    n_linhas = generate_txt(pessoas_para_bbox, cam_obj, filepath_txt)
