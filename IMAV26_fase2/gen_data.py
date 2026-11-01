"""
generate_yolo_heatpoint_dataset.py

Gera um dataset sintético (imagens + labels YOLO) para detecção de
heatpoints, usando o Blender como motor de renderização.

IDEIA GERAL
-----------
1. Cada imagem de fundo (PNG) vira o "chão" de uma cena 3D: um plano
   texturizado com a foto, convertido para tons de cinza (RGB->BW) e
   ligado a um shader Emission — o render reproduz a intensidade da
   imagem diretamente, sem depender de iluminação da cena.
2. "Heatpoints" são discos pequenos, também com shader Emission, com
   intensidade (brilho) variável simulando temperatura, espalhados em
   posições aleatórias dentro do campo de visão da câmera.
3. A câmera fica posicionada acima do plano simulando a altitude de voo
   do drone, com o mesmo FOV horizontal usado no nó de detecção ROS
   (mesma matemática de GSD).
4. Depois do render, pós-processo com OpenCV/numpy para gerar GRAYSCALE
   (com ruído sintético de sensor: gaussiano + dead pixels + banding)
   e/ou INFERNO (colormap em cima do grayscale) — as duas saem da MESMA
   geometria, então o mesmo conjunto de labels YOLO vale pras duas.

ESTRUTURA DE SAÍDA (padrão esperado pelo Ultralytics/YOLO)
------------------------------------------------------------
Se você gerar só um modo (output_modes = ("gray",) por exemplo):

    <output_dir>/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt
      labels/val/*.txt
      data.yaml

Se gerar mais de um modo simultaneamente, cada modo vira um dataset
completo e independente (sufixo no nome da pasta), pois as imagens
grayscale e inferno não podem coexistir no mesmo `images/train`:

    <output_dir>_gray/images/{train,val}, labels/{train,val}, data.yaml
    <output_dir>_inferno/images/{train,val}, labels/{train,val}, data.yaml

OTIMIZAÇÕES DE PERFORMANCE (por que estava lento)
--------------------------------------------------
A versão anterior chamava `read_factory_settings` e recriava toda a
geometria/material do zero a CADA imagem — isso é overhead de Python/API
do Blender (single-thread, síncrono), não trabalho de shading, por isso
CPU e GPU ficavam ociosos enquanto o tempo ia todo em reconstrução de cena.
Aqui:
  - a cena é montada UMA VEZ; a cada amostra só atualizamos propriedades
    (posição de câmera, imagem de fundo, posição/tamanho/intensidade dos
    heatpoints) em vez de recriar objetos e materiais;
  - os heatpoints usam um "pool" de objetos pré-criados (reaproveitados
    via hide_render, em vez de criar/destruir a cada amostra);
  - o plano de fundo reaproveita as imagens já carregadas em memória
    (bpy.data.images) em vez de recarregar o PNG do disco a cada amostra;
  - `scene.render.use_persistent_data = True` reaproveita dados
    derivados entre renders sucessivos na mesma sessão;
  - efeitos do Eevee que não fazem sentido numa cena 100% emissiva e sem
    luzes (AO, bloom, SSR, sombras) são desligados;
  - amostras de anti-aliasing reduzidas (cena é toda "flat", não precisa
    de muitas amostras para convergir).

COMO RODAR
----------
    blender --background --python gen_data.py -- \
    --backgrounds_dir ~/datasets/IMAV26_fase2/floor \
    --output_dir ~/datasets/IMAV26_fase2/

    É NECESSÁRIO TER OPENCV INSTALADO NO SEU SISTEMA
    OU INTERPRETADOR PYTHON QUE O BLENDER UTILIZA!!!!
    """

import glob
import math
import os
import random
import sys

import bpy
import bmesh
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

try:
    import cv2
except ImportError:
    cv2 = None
    print("[AVISO] cv2 não encontrado no Python do Blender. Instale com:\n"
          "  <caminho_do_blender>/python/bin/python3.x -m pip install opencv-python-headless")


# =============================================================================
# CONFIG — ajuste aqui
# =============================================================================
CONFIG = {
    # --- entrada/saída ---
    "backgrounds_dir": "/home/joao/dataset/backgrounds/",
    "output_dir": "/home/joao/dataset/dataset/",
    "images_per_background": 1,
    "val_split": 0.15,

    # --- câmera / voo ---
    "altitude_range_m": (30.0, 45.0),
    "camera_hfov_deg": 56.0,
    "resolution": (640, 512),
    "random_yaw": True,

    # --- heatpoints ---
    "heatpoints_per_image": (0, 4),      # (min, max) -- max também define o pool
    "prob_zero_heatpoints": 0.30,
    # Referência: drone a ~40m de altura vendo algo de no máximo ~3m de
    # largura. Variação moderada pra mais/menos, sem ir a extremos (nada
    # de pontos minúsculos de 0.2m nem muito maiores que 3m).
    "heatpoint_size_range_m": (1.2, 3.0),
    # área mínima ABSOLUTA (em pixels) de um blob na máscara -- só pra
    # descartar ruído/serrilhado de anti-aliasing, não pra decidir se o
    # heatpoint "conta" (isso é o heatpoint_min_visible_frac abaixo).
    "heatpoint_min_visible_px": 3,
    # área mínima RELATIVA: fração da área esperada do heatpoint (calculada
    # pelo tamanho real em metros x pixels/metro daquela amostra) que
    # precisa estar visível na máscara pra virar label. Sem isso, um
    # heatpoint 95% cortado pela borda da câmera ou 95% coberto por uma
    # mancha ainda passava (sobrava só uma lasca de poucos pixels, mas
    # acima do piso absoluto) -- ver match_visible_heatpoints.
    "heatpoint_min_visible_frac": 0.35,
    # quantas vezes tenta resortear posições quando TODOS os heatpoints
    # pretendidos (n>0) somem por acaso (ocluídos/cortados) -- sem isso,
    # esses casos viram "imagem vazia" sem querer, e a taxa real de
    # imagens sem objeto passa a ficar bem acima de prob_zero_heatpoints.
    "heatpoint_resample_attempts": 4,
    # obs: o piso foi subido de 0.6 pra 0.82 pra sempre ficar acima do teto
    # de brilho do fundo (background_max_intensity_frac) -- ver mais abaixo.
    "heatpoint_intensity_range": (0.82, 1.0),
    "heatpoint_margin_frac": 0.08,

    # --- manchas escuras (distratores "hard negative") ---
    # A maioria das manchas usa formas ORGÂNICAS (blobs irregulares, tipo
    # mancha/tinta derramada -- ver mancha_organic_*); só uma minoria usa
    # as formas "limpas" (heatpoint_shape_types) pra dar alguma variedade
    # geométrica também. Não entram no label YOLO -- servem só pra
    # ensinar a rede a não confundir blob escuro/textura com heatpoint.
    # Posições são independentes das dos heatpoints, então ocasionalmente
    # uma mancha cai em cima (ou embaixo) de um heatpoint, gerando oclusão
    # parcial -- ver SceneRig.apply_sample.
    "manchas_per_image": (0, 6),
    "prob_zero_manchas": 0.25,
    "mancha_size_range_m": (0.8, 12.0),
    "mancha_intensity_range": (0.0, 0.18),
    "mancha_margin_frac": -0.25,   # negativo = mancha pode nascer cortada na borda
    "mancha_occlude_prob": 0.35,   # chance da mancha renderizar NA FRENTE do heatpoint
    "mancha_organic_shape_count": 8,   # quantas variações de blob pré-gerar
    "mancha_organic_roughness": 0.45,  # 0 = quase um círculo, 1 = bem irregular
    "mancha_organic_prob": 0.8,        # fração das manchas que usa forma orgânica

    # --- variedade de forma dos heatpoints (formas "limpas"; manchas
    # também sorteiam destas formas na minoria dos casos -- ver acima) ---
    "heatpoint_shape_types": ("circle", "square", "rounded_square", "rounded_square_soft"),
    "shape_aspect_ratio_range": (0.4, 1.0),  # retângulos: razão entre os dois lados
    "shape_rotation": True,

    # --- contraste do fundo ---
    # Sem isso, fundos muito claros (céu, concreto estourado) geram
    # emissão perto de 1.0 -- no colormap INFERNO isso vira amarelo/branco,
    # a mesma faixa de cor de um heatpoint "quente", e a rede perde o
    # contraste que deveria aprender. Aqui o brilho do fundo é multiplicado
    # por esse fator ANTES da emissão, reservando a faixa alta (acima do
    # piso de heatpoint_intensity_range) só pra heatpoints de verdade.
    "background_max_intensity_frac": 0.72,

    # --- blur ocasional (menos contraste) ---
    # Em vez de estourar sempre com bordas 100% nítidas, uma fração das
    # imagens sai com os heatpoints desfocados, e outra fração (independente)
    # sai com as manchas desfocadas -- ver blur_regions/save_outputs.
    "heatpoint_blur_prob": 0.35,
    "mancha_blur_prob": 0.35,
    "blur_kernel_sizes": (5, 7, 9, 11, 13),

    # --- render ---
    "render_engine": "BLENDER_EEVEE",   # ou "BLENDER_EEVEE_NEXT" em Blender 4.2+
    "samples": 6,                        # cena flat/emissiva não precisa de muito

    # --- pós-processamento / ruído sintético ---
    "output_modes": ["inferno"],
    "gaussian_noise_std": 6.0,
    "dead_pixel_prob": 0.0005,
    "banding_strength": 4.0,

    "seed": 42,
}
# =============================================================================


def set_config_from_cli():
    if "--" not in sys.argv:
        return
    argv = sys.argv[sys.argv.index("--") + 1:]
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key in CONFIG and i + 1 < len(argv):
            current = CONFIG[key]
            value = argv[i + 1]
            if isinstance(current, tuple):
                nums = [float(v) if "." in v else int(v) for v in value.split(",")]
                CONFIG[key] = tuple(nums)
            elif isinstance(current, bool):
                CONFIG[key] = value.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                CONFIG[key] = int(value)
            elif isinstance(current, float):
                CONFIG[key] = float(value)
            else:
                CONFIG[key] = value
            i += 2
        else:
            i += 1


def ground_footprint_m(altitude_m, hfov_deg, resolution):
    hfov_rad = math.radians(hfov_deg)
    width_m = 2.0 * altitude_m * math.tan(hfov_rad / 2.0)
    aspect = resolution[1] / resolution[0]
    height_m = width_m * aspect
    return width_m, height_m


# -----------------------------------------------------------------------------
# Setup de cena — chamado UMA VEZ
# -----------------------------------------------------------------------------
class SceneRig:
    """Guarda referências reutilizáveis (câmera, plano de fundo, pool de
    heatpoints, cache de imagens) para evitar recriar tudo a cada amostra."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.scene = self._setup_scene()
        self.cam_obj = self._create_camera()
        self.plane, self.plane_tex_node = self._create_background_plane()
        self.image_cache = {}  # path -> bpy.data.images datablock

        self.shape_meshes = self._create_shape_meshes()
        default_mesh = self.shape_meshes["circle"]
        max_heatpoints = cfg["heatpoints_per_image"][1]
        max_manchas = cfg["manchas_per_image"][1]
        self.heatpoint_pool = self._create_object_pool(max_heatpoints, "Heatpoint", default_mesh)
        self.mancha_pool = self._create_object_pool(max_manchas, "Mancha", default_mesh)

    # ------------------------------------------------------------------
    def _setup_scene(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)  # só roda 1x, no início
        scene = bpy.context.scene
        cfg = self.cfg

        scene.render.engine = cfg["render_engine"]
        scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "BW"
        scene.render.use_persistent_data = True  # reaproveita dados entre renders

        # cor: transform "Standard" evita curva Filmic distorcer os valores
        # de intensidade que representam "temperatura" no nosso caso
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"

        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = cfg["samples"]
            # desliga efeitos irrelevantes p/ cena 100% emissiva sem luzes
            for attr, val in [
                ("use_gtao", False),
                ("use_bloom", False),
                ("use_ssr", False),
                ("use_soft_shadows", False),
                ("use_volumetric_lights", False),
                ("use_shadows", False),
            ]:
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, val)

        scene.render.use_motion_blur = False
        return scene

    # ------------------------------------------------------------------
    def _create_camera(self):
        cfg = self.cfg
        bpy.ops.object.camera_add(location=(0.0, 0.0, cfg["altitude_range_m"][0]))
        cam_obj = bpy.context.object
        cam_obj.data.lens_unit = "FOV"
        cam_obj.data.sensor_fit = "HORIZONTAL"
        cam_obj.data.angle = math.radians(cfg["camera_hfov_deg"])
        self_scene = bpy.context.scene
        self_scene.camera = cam_obj
        return cam_obj

    # ------------------------------------------------------------------
    def _create_background_plane(self):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.object
        plane.name = "BackgroundPlane"

        mat = bpy.data.materials.new(name="BackgroundMat")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emission = nt.nodes.new("ShaderNodeEmission")
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.extension = "REPEAT"

        # -- correção do problema "margem preta ao redor do fundo" --
        # O plano de fundo tem tamanho EXATO do footprint (width_m x
        # height_m) só quando a câmera olha reto pra baixo (yaw=0). Com
        # yaw != 0 o retângulo que a câmera realmente enxerga gira, e os
        # cantos do quadro passam a cair FORA desse retângulo -- aparece
        # preto ali. A correção: o plano vira um QUADRADO do tamanho da
        # DIAGONAL do footprint (grande o bastante pra conter o footprint
        # girado em qualquer ângulo -- ver apply_sample), e a imagem passa
        # a REPETIR (tile) sobre essa área extra via UVMap -> Mapping,
        # em vez de esticar uma cópia só e deixar buraco preto na sobra.
        # tile_x/tile_y = quantas vezes a imagem se repete ao longo de
        # cada eixo do plano ampliado; são calculados 1x aqui a partir do
        # aspect ratio da resolução (não dependem da altitude/amostra).
        aspect = self.cfg["resolution"][1] / self.cfg["resolution"][0]  # height/width
        diag_factor = math.hypot(1.0, aspect)  # diagonal / width, p/ footprint width=1
        tile_x = diag_factor
        tile_y = diag_factor / aspect

        uv_map = nt.nodes.new("ShaderNodeUVMap")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (tile_x, tile_y, 1.0)

        # -- correção do problema "fundo claro -> tudo amarelo no INFERNO" --
        # Sem isso, um fundo quase branco (concreto/céu estourado) gera
        # emissão perto de 1.0, na mesma faixa de um heatpoint "quente" --
        # no INFERNO os dois viram a mesma cor amarela/branca e a rede
        # perde o contraste que deveria aprender. Este nó Math multiplica
        # o brilho do fundo por um teto fixo (background_max_intensity_frac),
        # reservando a faixa alta de brilho só pra heatpoints de verdade.
        bg_cap = nt.nodes.new("ShaderNodeMath")
        bg_cap.operation = "MULTIPLY"
        bg_cap.inputs[1].default_value = self.cfg["background_max_intensity_frac"]

        nt.links.new(uv_map.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
        nt.links.new(bw.outputs["Val"], bg_cap.inputs[0])
        nt.links.new(bg_cap.outputs["Value"], emission.inputs["Color"])
        emission.inputs["Strength"].default_value = 1.0
        nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])

        plane.data.materials.append(mat)
        return plane, tex  # guardamos o nó de textura p/ trocar a imagem depois

    # ------------------------------------------------------------------
    @staticmethod
    def _build_rounded_rect_mesh(name, corner_radius, segments_per_corner=6):
        """Constrói (sem bevel) um retângulo unitário (extents -1..1) com
        cantos arredondados, gerando o contorno ponto a ponto -- 4 arcos
        de canto concatenados. corner_radius=0 dá um quadrado perfeito;
        perto de 1.0 fica quase um círculo. Construir os vértices na mão
        (em vez de usar bmesh.ops.bevel numa face de 4 vértices) evita o
        artefato de geometria degenerada que aparecia como uma "cruz"."""
        r = max(0.0, min(corner_radius, 1.0))
        corners = [
            (1.0 - r, 1.0 - r, 0.0),
            (-(1.0 - r), 1.0 - r, 90.0),
            (-(1.0 - r), -(1.0 - r), 180.0),
            (1.0 - r, -(1.0 - r), 270.0),
        ]
        verts = []
        for cx, cy, start_deg in corners:
            for i in range(segments_per_corner + 1):
                ang = math.radians(start_deg + 90.0 * i / segments_per_corner)
                verts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), 0.0))

        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bm_verts = [bm.verts.new(v) for v in verts]
        bm.faces.new(bm_verts)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    @staticmethod
    def _build_blob_mesh(name, roughness, seed):
        """Mancha orgânica: um polígono fechado com raio variando
        aleatoriamente por ângulo (suavizado pra não virar estrela
        picotada), lembrando uma mancha/respingo em vez de uma forma
        geométrica limpa. `seed` usa um random.Random próprio, então não
        mexe no estado do `random` global (mantém reprodutibilidade do
        resto do dataset)."""
        rng = random.Random(seed)
        num_points = rng.randint(10, 16)
        radii = [1.0 + rng.uniform(-roughness, roughness) for _ in range(num_points)]
        for _ in range(2):  # suaviza a variação de raio entre vizinhos
            radii = [
                (radii[i - 1] + radii[i] + radii[(i + 1) % num_points]) / 3.0
                for i in range(num_points)
            ]
        max_r = max(radii)
        radii = [r / max_r for r in radii]  # normaliza p/ caber em extents -1..1

        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bm_verts = []
        for i, r in enumerate(radii):
            ang = 2 * math.pi * i / num_points
            bm_verts.append(bm.verts.new((r * math.cos(ang), r * math.sin(ang), 0.0)))
        bm.faces.new(bm_verts)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    def _create_shape_meshes(self):
        """Cria (uma vez) as malhas unitárias usadas pelos heatpoints
        (formas "limpas": círculo, quadrado, arredondado) e pelas manchas
        (as mesmas formas limpas + várias variações de blob orgânico).
        Todas ficam centradas na origem com meia-extensão 1.0, então o
        tamanho final em metros é controlado só via object.scale. Trocar
        de forma vira só uma troca de referência (obj.data = ...), barata,
        sem recriar geometria a cada amostra."""
        cfg = self.cfg
        meshes = {}

        # círculo
        mesh = bpy.data.meshes.new("ShapeCircle")
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, radius=1.0, segments=20)
        bm.to_mesh(mesh)
        bm.free()
        meshes["circle"] = mesh

        # quadrado (também funciona como retângulo via scale não-uniforme)
        meshes["square"] = self._build_rounded_rect_mesh("ShapeSquare", corner_radius=0.0)
        # quadrado com cantos bem arredondados (quase um "squircle")
        meshes["rounded_square"] = self._build_rounded_rect_mesh("ShapeRoundedSquare", corner_radius=0.55)
        # quadrado com cantos levemente arredondados (variação mais sutil)
        meshes["rounded_square_soft"] = self._build_rounded_rect_mesh("ShapeRoundedSquareSoft", corner_radius=0.25)

        # manchas orgânicas -- várias variações pré-geradas com seed fixa
        for i in range(cfg["mancha_organic_shape_count"]):
            meshes[f"blob_{i}"] = self._build_blob_mesh(
                f"ShapeBlob_{i}", cfg["mancha_organic_roughness"], seed=cfg["seed"] * 1000 + i
            )

        return meshes

    # ------------------------------------------------------------------
    def _create_object_pool(self, count, name_prefix, default_mesh):
        """Cria um pool genérico de objetos reutilizáveis (usado tanto
        pros heatpoints quanto pras manchas). Cada objeto tem seu próprio
        material (criado 1x) cujo Strength/Color é só atualizado a cada
        amostra; a malha (forma) é trocada por referência em apply_sample,
        sem recriar geometria."""
        pool = []
        for i in range(count):
            obj = bpy.data.objects.new(f"{name_prefix}_{i}", default_mesh)
            bpy.context.collection.objects.link(obj)
            obj.hide_render = True

            mat = bpy.data.materials.new(name=f"{name_prefix}Mat_{i}")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            emission = nt.nodes.new("ShaderNodeEmission")
            emission.inputs["Strength"].default_value = 1.0
            nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])
            obj.data.materials.append(mat)

            pool.append({"obj": obj, "emission": emission})
        return pool

    # ------------------------------------------------------------------
    def get_background_image(self, path):
        """Carrega o PNG uma única vez e reaproveita o datablock em
        memória nas chamadas seguintes (em vez de recarregar do disco)."""
        img = self.image_cache.get(path)
        if img is None:
            img = bpy.data.images.load(path)
            self.image_cache[path] = img
        return img

    # ------------------------------------------------------------------
    def _apply_shape_specs(self, pool, specs, z_offset, yaw_rad=0.0):
        """Aplica uma lista de specs (posição, forma, escala, rotação,
        intensidade) num pool de objetos reutilizáveis. As posições x,y
        das specs são relativas ao referencial da câmera (sem rotação);
        aqui giramos elas pelo mesmo yaw da câmera antes de virar
        posição no mundo, senão os objetos ficam desalinhados do que a
        câmera realmente enxerga quando yaw != 0. Trocar a malha
        (obj.data) é só uma troca de referência -- barato, não recria
        geometria."""
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        for i, slot in enumerate(pool):
            if i < len(specs):
                spec = specs[i]
                obj = slot["obj"]
                mesh = self.shape_meshes[spec["shape"]]
                if obj.data is not mesh:
                    obj.data = mesh
                lx, ly = spec["x"], spec["y"]
                wx = lx * cos_y - ly * sin_y
                wy = lx * sin_y + ly * cos_y
                obj.location = (wx, wy, z_offset)
                obj.scale = (spec["rx"], spec["ry"], 1.0)
                obj.rotation_euler = (0.0, 0.0, spec["rot"])
                obj.hide_render = False
                v = spec["intensity"]
                slot["emission"].inputs["Color"].default_value = (v, v, v, 1.0)
            else:
                slot["obj"].hide_render = True

    def apply_sample(self, background_path, altitude_m, yaw_rad, heatpoint_specs, mancha_specs):
        """Atualiza a cena para a amostra atual (sem recriar nada)."""
        cfg = self.cfg
        width_m, height_m = ground_footprint_m(altitude_m, cfg["camera_hfov_deg"], cfg["resolution"])

        self.cam_obj.location = (0.0, 0.0, altitude_m)
        self.cam_obj.rotation_euler = (0.0, 0.0, yaw_rad)

        # O plano vira um QUADRADO do tamanho da diagonal do footprint --
        # grande o bastante pra cobrir o quadro da câmera girado em
        # qualquer yaw (ver comentário em _create_background_plane). A
        # textura repete (tile) sobre essa área via o node Mapping, que
        # já foi configurado 1x na criação do material.
        diag_m = math.hypot(width_m, height_m)
        self.plane.scale = (diag_m, diag_m, 1.0)
        self.plane_tex_node.image = self.get_background_image(background_path)

        # Heatpoint fica numa altura Z fixa; a mancha ora renderiza um
        # pouco ABAIXO (fica "atrás", só ao lado do heatpoint quando
        # sobrepõe), ora um pouco ACIMA (ocluindo parcialmente o
        # heatpoint por baixo) -- como as posições de mancha e heatpoint
        # são sorteadas de forma independente, a sobreposição -- e
        # portanto a oclusão -- acontece só ocasionalmente, por acaso.
        heatpoint_z = 0.002
        manchas_on_top = random.random() < cfg["mancha_occlude_prob"]
        mancha_z = 0.0025 if manchas_on_top else 0.0015

        self._apply_shape_specs(self.mancha_pool, mancha_specs, mancha_z, yaw_rad)
        self._apply_shape_specs(self.heatpoint_pool, heatpoint_specs, heatpoint_z, yaw_rad)

        return width_m, height_m


# -----------------------------------------------------------------------------
# Visibilidade real: renderiza uma máscara auxiliar em vez de inferir
# -----------------------------------------------------------------------------
# A tentativa anterior (checar oclusão via ray_cast na geometria) não foi
# suficiente -- entre outras coisas, o ray_cast do Blender testa contra o
# depsgraph "de viewport", que não necessariamente respeita hide_render
# nos objetos do pool que estão fora de uso naquela amostra, e ainda
# assim é só uma aproximação por amostragem de pontos. A solução direta
# é deixar o próprio Blender renderizar a verdade: uma passada auxiliar,
# rápida (samples=1, cena 100% flat), onde:
#   - o plano de fundo fica oculto (hide_render) -- não pode virar "blob"
#   - cada heatpoint ATIVO vira emissão branca pura (1,1,1)
#   - cada mancha ATIVA vira preto opaco (0,0,0) -- continua bloqueando
#     visualmente o que estiver atrás/abaixo dela (é um sólido, não
#     precisa de emissão pra ocluir), mas não aparece como blob "quente"
# O resultado é uma imagem binária: branco = exatamente o que vai
# aparecer como heatpoint na imagem final (já considerando oclusão por
# manchas, corte na borda da câmera, etc.). Os labels YOLO saem daí via
# connected components -- não mais da projeção 3D da geometria.
def render_visibility_mask(rig, tmp_dir, base_name):
    scene = rig.scene

    orig_bg_hide = rig.plane.hide_render
    rig.plane.hide_render = True

    orig_heat_colors = []
    for slot in rig.heatpoint_pool:
        orig_heat_colors.append(tuple(slot["emission"].inputs["Color"].default_value))
        if not slot["obj"].hide_render:
            slot["emission"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    orig_mancha_colors = []
    for slot in rig.mancha_pool:
        orig_mancha_colors.append(tuple(slot["emission"].inputs["Color"].default_value))
        if not slot["obj"].hide_render:
            slot["emission"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    orig_samples = None
    if hasattr(scene, "eevee"):
        orig_samples = scene.eevee.taa_render_samples
        scene.eevee.taa_render_samples = 1  # máscara binária não precisa de AA

    mask_path = os.path.join(tmp_dir, base_name + "_mask.png")
    scene.render.filepath = mask_path
    bpy.ops.render.render(write_still=True)

    # -- restaura o estado original da cena pra continuar reaproveitando
    # o rig normalmente nas próximas amostras --
    rig.plane.hide_render = orig_bg_hide
    for slot, color in zip(rig.heatpoint_pool, orig_heat_colors):
        slot["emission"].inputs["Color"].default_value = color
    for slot, color in zip(rig.mancha_pool, orig_mancha_colors):
        slot["emission"].inputs["Color"].default_value = color
    if orig_samples is not None:
        scene.eevee.taa_render_samples = orig_samples

    return mask_path


def components_from_visibility_mask(mask_path, min_area_px):
    """Lê a máscara branco/preto e devolve os blobs brancos conectados
    (posição, área, centróide). Só filtra ruído de anti-aliasing aqui
    (min_area_px baixo) -- a decisão de "esse heatpoint conta ou não"
    é feita depois, em match_visible_heatpoints, comparando a área do
    blob com a área que aquele heatpoint deveria ter."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n_labels, _labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    comps = []
    for i in range(1, n_labels):  # índice 0 é o fundo
        x, y, w, h, area = stats[i]
        if area < min_area_px or w <= 0 or h <= 0:
            continue
        comps.append({
            "bbox_px": (x, y, x + w, y + h),
            "area": float(area),
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
        })
    return comps


def project_point_px(scene, cam_obj, world_point, resolution):
    """Projeta um ponto 3D pra coordenada de pixel (mesma convenção X/Y
    da imagem renderizada). Retorna None se o ponto está atrás da câmera."""
    ndc = world_to_camera_view(scene, cam_obj, world_point)
    if ndc.z <= 0:
        return None
    return ndc.x * resolution[0], (1.0 - ndc.y) * resolution[1]


def expected_area_px(spec, px_per_m):
    """Área "cheia" esperada do heatpoint em pixels, a partir do tamanho
    real (metros) e da escala pixels/metro da amostra -- referência pra
    decidir se o blob que sobrou na máscara é o heatpoint quase inteiro
    ou só uma lasca residual (cortada pela borda da câmera ou coberta
    por uma mancha)."""
    rx_px = spec["rx"] * px_per_m
    ry_px = spec["ry"] * px_per_m
    if spec["shape"] == "circle":
        return math.pi * rx_px * ry_px
    # quadrado/arredondado -- área do retângulo é um limite superior
    # seguro mesmo pros cantos arredondados
    return 4.0 * rx_px * ry_px


def match_visible_heatpoints(rig, heatpoint_specs, components, scene, cam_obj,
                              resolution, px_per_m, min_visible_frac):
    """Casa cada heatpoint ATIVO com o blob branco mais próximo do seu
    centro projetado, e só aceita como label se a área do blob for pelo
    menos `min_visible_frac` da área esperada (cheia) daquele heatpoint.
    Isso descarta tanto heatpoints quase todo cobertos por uma mancha
    quanto heatpoints quase todo cortados pela borda do quadro -- os
    dois casos deixam só uma lasca de poucos pixels na máscara, que
    passava no filtro antigo (só de área mínima absoluta)."""
    remaining = list(components)
    label_lines, bboxes_px = [], []
    W, H = resolution

    for i, spec in enumerate(heatpoint_specs):
        obj = rig.heatpoint_pool[i]["obj"]
        if obj.hide_render:
            continue

        center_px = project_point_px(scene, cam_obj, obj.matrix_world.translation, resolution)
        if center_px is None:
            continue

        expected_area = expected_area_px(spec, px_per_m)
        if expected_area <= 0:
            continue

        max_dist = max(spec["rx"], spec["ry"]) * px_per_m * 2.0 + 5.0
        best, best_dist = None, None
        for comp in remaining:
            dx = comp["centroid"][0] - center_px[0]
            dy = comp["centroid"][1] - center_px[1]
            dist = math.hypot(dx, dy)
            if dist <= max_dist and (best is None or dist < best_dist):
                best, best_dist = comp, dist

        if best is None:
            continue  # nada sobrou visível pra esse heatpoint

        remaining.remove(best)
        if best["area"] / expected_area < min_visible_frac:
            continue  # só uma lasca residual -- descarta

        x0, y0, x1, y1 = best["bbox_px"]
        cx = (x0 + x1) / 2.0 / W
        cy = (y0 + y1) / 2.0 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        bboxes_px.append((x0, y0, x1, y1))

    return label_lines, bboxes_px


# -----------------------------------------------------------------------------
# Bounding boxes (projeção 3D -> 2D) -- ainda usada só pra saber ONDE
# aplicar blur nas manchas (mancha_bboxes_px); os labels de heatpoint
# agora vêm de match_visible_heatpoints, não mais daqui.
# -----------------------------------------------------------------------------
def object_bbox_yolo(scene, cam_obj, obj, resolution):
    world_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]

    xs, ys = [], []
    for corner in world_corners:
        co_ndc = world_to_camera_view(scene, cam_obj, corner)
        if co_ndc.z <= 0:
            continue
        xs.append(co_ndc.x)
        ys.append(1.0 - co_ndc.y)

    if not xs or not ys:
        return None

    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))

    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return None
    if w * resolution[0] < 1 or h * resolution[1] < 1:
        return None

    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, w, h)


# -----------------------------------------------------------------------------
# Pós-processamento (ruído sintético + colormaps)
# -----------------------------------------------------------------------------
def add_sensor_noise(gray_img, gaussian_std, dead_pixel_prob, banding_strength):
    img = gray_img.astype(np.float32)

    if gaussian_std > 0:
        img += np.random.normal(0, gaussian_std, img.shape)

    if banding_strength > 0:
        h = img.shape[0]
        banding = np.random.normal(0, banding_strength, size=(h, 1))
        img += banding

    if dead_pixel_prob > 0:
        mask = np.random.random(img.shape) < dead_pixel_prob
        dead_values = np.random.choice([0, 255], size=mask.sum())
        img[mask] = dead_values

    return np.clip(img, 0, 255).astype(np.uint8)


def bbox_norm_to_px(cx, cy, w, h, resolution):
    W, H = resolution
    x0 = int(round((cx - w / 2.0) * W))
    y0 = int(round((cy - h / 2.0) * H))
    x1 = int(round((cx + w / 2.0) * W))
    y1 = int(round((cy + h / 2.0) * H))
    return x0, y0, x1, y1


def blur_regions(img_f32, bboxes_px, kernel_sizes):
    """Aplica um blur (kernel aleatório por região) só DENTRO de cada
    bbox (com uma margem ao redor e borda suavizada por uma máscara
    gaussiana), pra simular desfoque óptico/de sensor sem borrar o resto
    da imagem. Usado tanto pra heatpoints quanto pra manchas -- ver
    save_outputs."""
    if not bboxes_px:
        return img_f32

    h_img, w_img = img_f32.shape[:2]
    out = img_f32.copy()

    for (x0, y0, x1, y1) in bboxes_px:
        pad_x = max(3, int((x1 - x0) * 0.6))
        pad_y = max(3, int((y1 - y0) * 0.6))
        rx0, ry0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        rx1, ry1 = min(w_img, x1 + pad_x), min(h_img, y1 + pad_y)
        if rx1 - rx0 < 3 or ry1 - ry0 < 3:
            continue

        roi = out[ry0:ry1, rx0:rx1]
        k = random.choice(kernel_sizes)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        # máscara: 1.0 na área original da bbox, some suavemente até 0
        # na borda do ROI ampliado -- evita um retângulo "duro" de blur
        mask = np.zeros(roi.shape, dtype=np.float32)
        core_x0, core_y0 = x0 - rx0, y0 - ry0
        core_x1, core_y1 = x1 - rx0, y1 - ry0
        mask[core_y0:core_y1, core_x0:core_x1] = 1.0
        feather = max(2.0, min(pad_x, pad_y) / 2.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
        mask = np.clip(mask, 0.0, 1.0)

        out[ry0:ry1, rx0:rx1] = roi * (1.0 - mask) + blurred * mask

    return out


def save_outputs(raw_render_path, out_paths, cfg, heatpoint_bboxes_px, mancha_bboxes_px):
    if cv2 is None:
        raise RuntimeError("cv2 não disponível no Python do Blender.")

    gray = cv2.imread(raw_render_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # blur ocasional e independente pra heatpoints e pra manchas -- só
    # dentro das respectivas regiões, pra tirar um pouco do contraste
    # "perfeito" de borda nítida sem borrar a imagem toda.
    if heatpoint_bboxes_px and random.random() < cfg["heatpoint_blur_prob"]:
        gray = blur_regions(gray, heatpoint_bboxes_px, cfg["blur_kernel_sizes"])
    if mancha_bboxes_px and random.random() < cfg["mancha_blur_prob"]:
        gray = blur_regions(gray, mancha_bboxes_px, cfg["blur_kernel_sizes"])

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = add_sensor_noise(
        gray, cfg["gaussian_noise_std"], cfg["dead_pixel_prob"], cfg["banding_strength"]
    )

    if "gray" in out_paths:
        cv2.imwrite(out_paths["gray"], gray)
    if "inferno" in out_paths:
        inferno = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        cv2.imwrite(out_paths["inferno"], inferno)


# -----------------------------------------------------------------------------
# Geração de uma amostra
# -----------------------------------------------------------------------------
def _sample_shape_spec(shape, x, y, size_m, intensity, cfg):
    """Monta uma spec de forma (usada tanto por heatpoints quanto por
    manchas) a partir de uma forma JÁ escolhida pelo chamador: um
    raio/meia-extensão base a partir do tamanho sorteado, uma razão de
    aspecto (pra virar retângulo em vez de sempre quadrado/círculo
    perfeito) e uma rotação opcional em Z."""
    aspect = random.uniform(*cfg["shape_aspect_ratio_range"])
    r = size_m / 2.0
    if random.random() < 0.5:
        rx, ry = r, r * aspect
    else:
        rx, ry = r * aspect, r
    rot = random.uniform(0, 2 * math.pi) if cfg["shape_rotation"] else 0.0
    return {"x": x, "y": y, "rx": rx, "ry": ry, "rot": rot, "shape": shape, "intensity": intensity}


def sample_heatpoints(cfg, width_m, height_m):
    n_min, n_max = cfg["heatpoints_per_image"]
    if random.random() < cfg["prob_zero_heatpoints"]:
        n = 0
    else:
        n = random.randint(max(1, n_min), n_max)

    margin = cfg["heatpoint_margin_frac"]
    x_range = (-width_m / 2 * (1 - margin), width_m / 2 * (1 - margin))
    y_range = (-height_m / 2 * (1 - margin), height_m / 2 * (1 - margin))

    specs = []
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        size = random.uniform(*cfg["heatpoint_size_range_m"])
        intensity = random.uniform(*cfg["heatpoint_intensity_range"])
        shape = random.choice(cfg["heatpoint_shape_types"])
        specs.append(_sample_shape_spec(shape, x, y, size, intensity, cfg))
    return specs


def sample_manchas(cfg, width_m, height_m):
    """Manchas escuras: a maioria usa forma ORGÂNICA (blob irregular,
    tipo mancha/respingo -- ver mancha_organic_prob); o resto sorteia
    entre as formas "limpas" dos heatpoints, pra dar alguma variedade
    geométrica também. Não geram bounding box -- são puramente
    distratores visuais (ver generate_sample, que só lê
    rig.heatpoint_pool pros labels)."""
    n_min, n_max = cfg["manchas_per_image"]
    if random.random() < cfg["prob_zero_manchas"]:
        n = 0
    else:
        n = random.randint(n_min, n_max)

    margin = cfg["mancha_margin_frac"]
    x_range = (-width_m / 2 * (1 - margin), width_m / 2 * (1 - margin))
    y_range = (-height_m / 2 * (1 - margin), height_m / 2 * (1 - margin))

    blob_names = [f"blob_{i}" for i in range(cfg["mancha_organic_shape_count"])]

    specs = []
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        size = random.uniform(*cfg["mancha_size_range_m"])
        intensity = random.uniform(*cfg["mancha_intensity_range"])
        if blob_names and random.random() < cfg["mancha_organic_prob"]:
            shape = random.choice(blob_names)
        else:
            shape = random.choice(cfg["heatpoint_shape_types"])
        specs.append(_sample_shape_spec(shape, x, y, size, intensity, cfg))
    return specs


def generate_sample(rig, background_path, sample_idx, split, cfg, paths, tmp_dir):
    altitude = random.uniform(*cfg["altitude_range_m"])
    yaw = random.uniform(0, 2 * math.pi) if cfg["random_yaw"] else 0.0

    # amostra provisória do footprint p/ decidir posições dos heatpoints/manchas
    width_m, height_m = ground_footprint_m(altitude, cfg["camera_hfov_deg"], cfg["resolution"])
    px_per_m = cfg["resolution"][0] / width_m

    heatpoint_specs = sample_heatpoints(cfg, width_m, height_m)
    mancha_specs = sample_manchas(cfg, width_m, height_m)
    wanted_nonzero = len(heatpoint_specs) > 0  # essa amostra NÃO devia ser "vazia"

    base_name = f"{os.path.splitext(os.path.basename(background_path))[0]}_{sample_idx:04d}"
    raw_path = os.path.join(tmp_dir, base_name + "_raw.png")

    max_attempts = cfg["heatpoint_resample_attempts"] if wanted_nonzero else 1
    label_lines, heatpoint_bboxes_px = [], []

    for attempt in range(max_attempts):
        rig.apply_sample(background_path, altitude, yaw, heatpoint_specs, mancha_specs)

        rig.scene.render.filepath = raw_path
        bpy.ops.render.render(write_still=True)

        # ground truth real: renderiza uma passada auxiliar só com
        # heatpoints (brancos) x manchas (pretas, ocluindo) x fundo
        # oculto, e casa cada blob branco que sobrou com o heatpoint que
        # o originou -- só vira label se a área visível for uma fração
        # relevante da área "cheia" esperada (ver match_visible_heatpoints).
        mask_path = render_visibility_mask(rig, tmp_dir, base_name)
        components = components_from_visibility_mask(mask_path, cfg["heatpoint_min_visible_px"])
        os.remove(mask_path)

        label_lines, heatpoint_bboxes_px = match_visible_heatpoints(
            rig, heatpoint_specs, components, rig.scene, rig.cam_obj,
            cfg["resolution"], px_per_m, cfg["heatpoint_min_visible_frac"],
        )

        if label_lines or not wanted_nonzero:
            break  # deu certo, ou a amostra era mesmo pra ser "vazia"

        # todos os heatpoints pretendidos sumiram por acaso (ocluídos
        # e/ou cortados na borda) -- resorteia posições e tenta de novo,
        # senão essa amostra vira "sem objeto" sem querer, inflando a
        # taxa real de imagens vazias acima de prob_zero_heatpoints
        heatpoint_specs = sample_heatpoints(cfg, width_m, height_m)
        while not heatpoint_specs:  # não deixa o retry sortear "zero" por acaso
            heatpoint_specs = sample_heatpoints(cfg, width_m, height_m)
        mancha_specs = sample_manchas(cfg, width_m, height_m)

    # bbox das manchas serve só pra saber ONDE aplicar blur -- nunca vira label
    mancha_bboxes_px = []
    for slot in rig.mancha_pool:
        if slot["obj"].hide_render:
            continue
        bbox = object_bbox_yolo(rig.scene, rig.cam_obj, slot["obj"], cfg["resolution"])
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        mancha_bboxes_px.append(bbox_norm_to_px(cx, cy, w, h, cfg["resolution"]))

    for mode in cfg["output_modes"]:
        label_path = os.path.join(paths[mode]["labels"][split], base_name + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    out_paths = {
        mode: os.path.join(paths[mode]["images"][split], base_name + ".png")
        for mode in cfg["output_modes"]
    }
    save_outputs(raw_path, out_paths, cfg, heatpoint_bboxes_px, mancha_bboxes_px)
    os.remove(raw_path)


# -----------------------------------------------------------------------------
# Setup de pastas + data.yaml
# -----------------------------------------------------------------------------
def setup_output_dirs(cfg):
    """Estrutura por modo:
        <root>/images/{train,val}
        <root>/labels/{train,val}
        <root>/data.yaml
    Onde <root> = output_dir (se só 1 modo) ou output_dir + "_<modo>"
    (se mais de 1 modo, já que não dá pra misturar grayscale e inferno
    no mesmo images/train)."""
    single_mode = len(cfg["output_modes"]) == 1
    paths = {}
    for mode in cfg["output_modes"]:
        root = cfg["output_dir"] if single_mode else f"{cfg['output_dir']}_{mode}"
        mode_paths = {"root": root, "images": {}, "labels": {}}
        for split in ("train", "val"):
            img_dir = os.path.join(root, "images", split)
            lbl_dir = os.path.join(root, "labels", split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            mode_paths["images"][split] = img_dir
            mode_paths["labels"][split] = lbl_dir
        paths[mode] = mode_paths

    tmp_dir = os.path.join(cfg["output_dir"], "_tmp_raw")
    os.makedirs(tmp_dir, exist_ok=True)
    return paths, tmp_dir


def write_data_yaml(paths):
    for mode, mode_paths in paths.items():
        root = mode_paths["root"]
        yaml_path = os.path.join(root, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(
                f"path: {root}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: 1\n"
                f"names: ['heatpoint']\n"
            )
        print(f"[OK] {yaml_path}")


def find_backgrounds(directory):
    """Lista imagens de fundo (png/jpg/jpeg, qualquer caixa), ignorando
    mapas de normais (ex.: Grass_01_Nrm.png), que não são texturas de cor."""
    exts = (".png", ".jpg", ".jpeg")
    skip_tokens = ("_nrm", "_normal", "_nor.", "_n.")
    found = []
    for name in sorted(os.listdir(os.path.expanduser(directory))):
        low = name.lower()
        if not low.endswith(exts):
            continue
        if any(tok in low for tok in skip_tokens):
            print(f"[INFO] ignorando mapa de normais: {name}")
            continue
        found.append(os.path.join(os.path.expanduser(directory), name))
    return found


# -----------------------------------------------------------------------------
def main():
    set_config_from_cli()
    cfg = CONFIG
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    backgrounds = find_backgrounds(cfg["backgrounds_dir"])
    if not backgrounds:
        raise RuntimeError(f"Nenhuma imagem (png/jpg/jpeg) encontrada em {cfg['backgrounds_dir']}")
    print(f"[INFO] {len(backgrounds)} imagens de fundo encontradas.")

    paths, tmp_dir = setup_output_dirs(cfg)
    rig = SceneRig(cfg)  # <-- cena montada UMA VEZ

    sample_idx = 0
    for bg_path in backgrounds:
        for _ in range(cfg["images_per_background"]):
            split = "val" if random.random() < cfg["val_split"] else "train"
            generate_sample(rig, bg_path, sample_idx, split, cfg, paths, tmp_dir)
            sample_idx += 1
            if sample_idx % 20 == 0:
                print(f"[INFO] {sample_idx} amostras geradas...")

    write_data_yaml(paths)

    if not os.listdir(tmp_dir):
        os.rmdir(tmp_dir)

    print(f"[OK] Dataset gerado — total: {sample_idx} amostras.")
    for mode, mode_paths in paths.items():
        print(f"     [{mode}] -> {mode_paths['root']}")


if __name__ == "__main__":
    main()
