# YOLO_Pose_synthethic_dataset_for_SAE2026-third_phase

Este repositório contém os scripts necessários + arquivos suplementares para geração e labelling automático de um dataset sintético com a finalidade do treinamento de uma YOLO-pose para leitura de manômetros da fase 3 da SAE26

# Requisitos

Obviamente, além do Python e do Blender, você deve ter instalada as bibliotecas *Numpy, Albumentations e OpenCV*. Caso rode os scripts dentro do Blender, será necessário instalar as bibliotecas diretamente no Python do Blender. Para isso eu recomendo seguir o tutorial abaixo:

```shell
https://blender.stackexchange.com/questions/56011/how-to-install-pip-for-blenders-bundled-python-and-use-it-to-install-packages/283142#283142
```

# Como rodar?

Você deve alterar em **ambos** os scripts o endereço do folder para onde o dataset será armazenado em sua máquina e a faixa de ID das imagens geradas (o número que elas começam e terminam).

É possível gerar o dataset de forma colaborativa, neste caso, particione em seções de ID das imagens e combine com o resto das pessoas o pedaço de cada um (Ex. Pessoa 1 faz do 0000 até o 0020, Pessoa 2 do 0021 até o 0100...), o tempo estimado para gerar 5000 imagens em minha máquina é de cerca de 53 minutos, ou aproximadamente 0.6s por imagem.

Para rodar o script no terminal, basta executar o seguinte comando dentro da pasta do executável do Blender:

```shell
blender.exe -b "caminho\para\manometro.blend" --python "caminho\para\blender_headless.py"
```
Caso rode em uma máquina Linux, apenas basta remover o .exe e escrever o caminho utilizando "```/```" e não "```\```", sem necessidade de estar no diretório do executável do Blender. Além disso tem que garantir permissão ao script Python:

```shell
chmod +x /caminho/para/o/script
```

Após gerar as imagens, execute o script "aplica_efeitos.py" para aplicar os efeitos de ruído, blur e motion blur e compressão.

A estrutura do projeto segue:

```shell

├── dataset/
│   ├── images/      # Imagens PNG renderizadas e processadas
│   └── labels/      # Arquivos .txt (YOLO format)
├── floor_texture/   # Texturas que serão aplicadas ao piso
├── scripts/
│   ├── blender_headless.py
│   └── aplica_efeitos.py
└── manometros.blend     # Arquivo mestre do Blender
```

# O que os códigos fazem?

**blender_headless.py**: O script segue o seguinte fluxo de trabalho:

- Aleatoriza uma altura pra câmera em uma determinada margem, depois gera dois ângulos aleatórios: um para definir um raio de rotação e outro define a rotação no plano xy com base no raio;

- Aleatoriza a posição do manômetro em um quadrado 2x2, garantindo que o empty mestre do manômetro está dentro do quadro de composição da câmera;

- Como a rotação do manômetro é 0 durante a segunda etapa, é calculado a diferença de posição dele no plano xy para a câmera e por meio do arctan, rotacionamos para tentar garantir que cerca de 80% das fotos tenham a orientação correta (definição de ângulo do Blender goat aqui);

- Depois aleatorizamos a posição das luzes, imagem de fundo, yada yada. Puxamos a bounding box e a posição dos keypoints por meio da localização de alguns empties, fazemos clamping na bbox para não ir além de 1 (ou menos que 0), verificamos a visibilidade dos pontos e por fim, montamos o txt. 

- No main, definimos uma variavel aleatória, que representa a chance de 20% de gerar uma imagem sem manômetro, de forma a gerar lixo para rede também. 

**aplica_efeitos.py**: Não tem muito o que falar aqui, é só um script simples que aplica os efeitos de ruído, blur (motion blur ou gaussian blur), e compressão de imagem. 

# Por que não fazer tudo em um script só?

Meu Blender crasha de vez em quando se eu tento rodar o opencv/albumentations direto. Se o seu rodar tranquilo, ai é resenha. 
