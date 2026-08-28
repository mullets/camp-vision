# CAMP Vision

Catalogação automática de acervos de arquitetura, a partir de pranchas
digitalizadas em TIFF, usando visão computacional (OpenCV), OCR
(Tesseract) e interpretação por IA (API da OpenAI).

> **Compatibilidade em primeiro lugar.** O CAMP Vision é pensado para
> rodar em qualquer Mac — incluindo modelos antigos sem GPU e sem
> instruções de CPU recentes, como o Mac Pro 5,1 (2010, Xeon
> Westmere) e o 6,1 (2013, Xeon Ivy Bridge-EP). Por isso a interface
> usa **Tkinter** (embutido no Python) em vez de Qt, e o OCR usa
> **Tesseract** em vez de PaddleOCR por padrão — veja a seção
> [Rodando em Macs antigos](#rodando-em-macs-antigos-mac-pro-51--61)
> para o porquê.

---

## Visão geral

O usuário informa um código de projeto e seleciona uma pasta contendo
milhares de arquivos de imagem digitalizados — **JPG é o formato
principal** (mais rápido de ler/processar, sem dependências extras de
decodificação); TIFF (inclusive multipágina), PNG, BMP e PDF também
são suportados. O CAMP Vision processa cada arquivo em lote, de forma
automática:

1. Abre a imagem e verifica a resolução.
2. Corrige orientação, melhora contraste e reduz ruído.
3. Localiza automaticamente o carimbo (sem assumir posição fixa —
   procura nos quatro cantos da prancha).
4. Executa OCR no carimbo (Tesseract, por padrão).
5. Envia o texto do OCR para interpretação por IA (OpenAI), extraindo
   metadados estruturados. Se a IA não estiver disponível, usa um
   extrator por regras como *fallback*.
6. Corrige nomes (arquitetos, cidades) por similaridade textual
   (`difflib`, biblioteca padrão), comparando com um banco de
   conhecimento SQLite que cresce a cada execução.
7. Classifica automaticamente o tipo da prancha (Planta, Corte,
   Fachada, Estrutura, Elétrica etc.).
8. **Renomeia o arquivo original** com um código único de
   arquivamento (ex.: `SB-P001.1-00001.tif`).
9. **Organiza o arquivo em pastas por ano/projeto**, seguindo a
   lógica de classificação orientada pela Resolução CONARQ/MGI nº
   56/2024 (diretrizes para arquivos de arquitetura).
10. Grava os metadados extraídos diretamente no arquivo (EXIF/IPTC),
    para busca pelo Spotlight, Adobe Bridge etc.
11. Gera miniatura JPG e salva o recorte do carimbo em PNG.
12. Exporta os resultados em CSV, XLSX e JSON.

Todo o processamento roda em paralelo (multithread) e pode ser
cancelado a qualquer momento pela interface gráfica.

---

## Estrutura do projeto

```
campvision/
├── app.py                     # Ponto de entrada da aplicação
├── config.py                  # Configurações persistentes (JSON)
├── requirements.txt            # Dependências essenciais (leves)
├── requirements-ml.txt         # Extras opcionais (ML, hardware recente)
├── interface/                  # Interface gráfica (Tkinter)
│   ├── janela_principal.py
│   ├── dialogo_configuracoes.py
│   └── temas.py
├── scanner/                    # Leitura, pré-processamento e orquestração
│   ├── leitor_imagem.py        # Leitura de TIFF, correção, contraste, ruído
│   ├── detector_carimbo.py     # Localização automática do carimbo (+ fábrica)
│   ├── pipeline.py             # Pipeline completo por arquivo
│   └── lote.py                 # Orquestração paralela do lote
├── ocr/
│   ├── base.py                 # Estrutura de resultado compartilhada
│   ├── motor.py                # Fábrica de estratégia (Tesseract/PaddleOCR)
│   ├── tesseract_ocr.py        # Motor padrão (leve, sem exigência de AVX)
│   └── paddle_ocr.py           # Motor opcional (requer requirements-ml.txt)
├── ai/
│   ├── interpretador.py        # Interpretação via API da OpenAI
│   └── fallback_regras.py      # Extração por regras (sem IA)
├── classificacao/
│   └── classificador.py        # Classificação automática do tipo (+ fábrica)
├── ml/                          # Treino/inferência de modelos (opcional)
│   ├── dataset_carimbo.py / treinar_carimbo.py / detector_carimbo_ml.py
│   └── dataset_classificacao.py / treinar_classificador.py / classificador_ml.py
├── exportacao/
│   └── exportador.py           # CSV / XLSX / JSON / miniaturas / carimbos
├── database/
│   ├── models.py                    # Modelos SQLAlchemy (inclui Projeto/Endereço de obra)
│   ├── repository.py                # Correção inteligente (difflib) + sequenciais
│   └── importar_conhecimento.py     # Alimenta o banco a partir de planilhas curadas e catalogações antigas
├── utils/
│   ├── logger.py                 # Sistema de logs (arquivo + console + GUI)
│   ├── tempo.py                  # Estimativa de tempo restante (ETA)
│   ├── renomeador.py             # Geração do código único de arquivamento
│   ├── arquivamento.py           # Organização em pastas ano/projeto (CONARQ)
│   └── metadados_exif.py         # Gravação de EXIF/IPTC via exiftool
├── logs/                         # Logs gerados em tempo de execução
└── tests/                        # Testes unitários e de integração
```

---

## Instalação

### Pré-requisitos

- macOS (Intel ou Apple Silicon — inclusive modelos antigos, ver seção
  específica abaixo)
- Python 3.10 ou superior (não é necessário 3.13 — quanto mais nova a
  versão exigida, menor a chance de haver wheels pré-compiladas para
  macOS antigos)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract):
  `brew install tesseract tesseract-lang` (o pacote `tesseract-lang`
  traz o idioma português)
- [ExifTool](https://exiftool.org/) (opcional, para gravar metadados
  pesquisáveis nos arquivos): `brew install exiftool`
- Se o Python foi instalado via Homebrew, garanta o Tk:
  `brew install python-tk`

### Passo a passo

```bash
# 1. Clone ou copie o projeto para sua máquina
cd campvision

# 2. Crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências essenciais
pip install --upgrade pip
pip install -r requirements.txt
```

> Isso já é suficiente para rodar o CAMP Vision completo — detecção
> de carimbo por heurística, OCR via Tesseract, classificação por
> regras. Só instale `requirements-ml.txt` se quiser usar modelos
> treinados (ver seção específica) e seu Mac tiver CPU/GPU recente o
> bastante.

### Configuração da chave de API (OpenAI)

A chave pode ser definida na tela **Configurações** do próprio
aplicativo, ou por variável de ambiente antes de iniciar:

```bash
export OPENAI_API_KEY="sua-chave-aqui"
```

Se nenhuma chave for configurada, o sistema funciona normalmente,
usando apenas OCR e o extrator de metadados por regras.

---

## Como executar

```bash
source .venv/bin/activate
python app.py
```

A janela principal permite:

- **Código do projeto**: sigla usada para renomear os arquivos (ex.:
  "SB" para "Sami Bussab") — sugerido automaticamente a partir do
  nome da pasta selecionada, mas editável.
- **Selecionar Pasta**: escolhe a pasta com os arquivos TIFF.
- **Processar**: inicia o processamento em lote.
- **Cancelar**: interrompe o processamento em andamento (os arquivos
  já processados não são perdidos).
- **Configurações**: idioma, pasta padrão, qualidade das miniaturas,
  motor de OCR, modelo de IA, chave da API, quantidade de threads,
  renomeação, organização em pastas, gravação de EXIF, tema
  claro/escuro.

Por padrão, os arquivos originais são **renomeados e movidos** para:

```
<pasta selecionada>/<ano>/<nome da pasta>/CODIGO-PRANCHA-00001.tif
```

E os relatórios de catalogação ficam em:

```
<pasta selecionada>/catalogacao_saida/
├── catalogacao.csv
├── catalogacao.xlsx
├── catalogacao.json
├── miniaturas/
└── carimbos/
```

As linhas do CSV/XLSX/JSON saem agrupadas por projeto (mesmo código
que nomeia a pasta de arquivamento, ex. "OCG-P0032") e, dentro de
cada projeto, ordenadas pelo número da folha lido do carimbo —
pranchas sem número legível ficam no fim do grupo.

---

## Rodando em Macs antigos (Mac Pro 5,1 / 6,1)

Se o app não abria ou travava antes, provavelmente era por causa de
duas dependências que exigem hardware relativamente recente:

- **PySide6/Qt**: builds recentes do Qt 6 costumam exigir macOS 11+
  para instalar/rodar.
- **PaddlePaddle** (usado pelo PaddleOCR): os pacotes pré-compilados
  geralmente exigem instruções de CPU **AVX/AVX2**. O Mac Pro 5,1
  (Xeon Westmere, 2010) não tem AVX; o 6,1 (Xeon Ivy Bridge-EP, 2013)
  tem AVX mas não AVX2. Isso faz o PaddlePaddle falhar na instalação
  ou travar com erro de "illegal instruction" em tempo de execução.

A configuração padrão do projeto já evita os dois problemas:

- **Interface em Tkinter** (parte da biblioteca padrão do Python, sem
  exigência de versão de macOS além da do próprio Python).
- **OCR via Tesseract** por padrão (sem exigência de AVX/AVX2).
- **Sem PyTorch/Ultralytics** no `requirements.txt` padrão — esses
  ficam em `requirements-ml.txt`, só necessários se você habilitar
  detecção/classificação por modelo treinado (não recomendado em
  hardware antigo).
- **Sem RapidFuzz nem pandas** — usa `difflib` e `csv`/`openpyxl`,
  todos sem extensões compiladas problemáticas.

### Se o `pip install` do Python 3.10+ ainda falhar

Em macOS muito antigo, mesmo o Homebrew pode não oferecer uma versão
de Python compatível. Alternativas:

1. **pyenv**: permite compilar uma versão específica do Python a
   partir do código-fonte, compatível com seu macOS exato:
   ```bash
   brew install pyenv
   pyenv install 3.10.14
   pyenv local 3.10.14
   ```
2. **python.org**: os instaladores oficiais às vezes suportam versões
   de macOS mais antigas que o Homebrew — verifique a página de
   downloads de versões antigas do Python.

### Se o `tesseract` do Homebrew não instalar

Verifique se o seu Homebrew ainda recebe atualizações para o macOS
instalado (`brew doctor`). Em macOS realmente antigo, pode ser
necessário usar uma versão mais antiga do Homebrew ("Homebrew
Legacy") ou instalar o Tesseract por outro gerenciador de pacotes
(ex.: MacPorts).

---

## Executando os testes

```bash
pytest tests/ -v
```

---

## Como gerar um aplicativo nativo para macOS (.app)

O projeto já inclui um arquivo de especificação para o PyInstaller.

```bash
source .venv/bin/activate
pip install pyinstaller

pyinstaller campvision.spec
```

O aplicativo gerado estará em `dist/CAMP Vision.app`.

### Gerando um instalador DMG

Com o `.app` já gerado em `dist/`, use o utilitário `create-dmg`
(instalável via Homebrew):

```bash
brew install create-dmg

create-dmg \
  --volname "CAMP Vision" \
  --window-size 600 400 \
  --icon-size 100 \
  --app-drop-link 450 200 \
  "dist/CAMP-Vision-Installer.dmg" \
  "dist/CAMP Vision.app"
```

---

## Detecção e classificação por modelo treinado (opcional, hardware recente)

> Requer `pip install -r requirements-ml.txt` — não recomendado em
> Macs antigos sem AVX2 (ver seção acima).

Por padrão, o CAMP Vision usa:
- **detecção de carimbo**: heurística por contornos (OpenCV), sem necessidade de treino;
- **classificação de tipo**: palavras-chave do OCR;
- **OCR**: Tesseract.

Essas estratégias já funcionam "out of the box" em qualquer Mac. Em
hardware mais novo (Apple Silicon, ou Intel com AVX2 + GPU), é
possível trocar por modelos treinados para maior precisão.

### 1. Detector de carimbo (YOLO)

Anote manualmente a posição do carimbo em algumas dezenas/centenas de
pranchas (ferramentas como [LabelImg](https://github.com/heartexlabs/labelImg)
ou [CVAT](https://www.cvat.ai/) exportando em "YOLO format" servem bem),
gerando pares `imagem.tif` + `imagem.txt` em uma mesma pasta.

```bash
python -m ml.treinar_carimbo \
    --anotacoes /caminho/para/anotacoes \
    --saida /caminho/para/dataset_preparado \
    --modelos-saida modelos/ \
    --epocas 150
```

O modelo final fica em `modelos/treino_carimbo/weights/best.pt`.

### 2. Classificador de tipo de prancha (ResNet18)

Organize miniaturas já catalogadas (ex.: as geradas em
`catalogacao_saida/miniaturas/`) em pastas por categoria:

```
dataset_tipos/
    Planta/
    Corte/
    Fachada/
    ...
```

```bash
python -m ml.treinar_classificador \
    --dataset /caminho/para/dataset_tipos \
    --saida modelos/classificador_tipo.pt \
    --epocas 20
```

### 3. Ativando os modelos treinados / o motor PaddleOCR

Na tela **Configurações**, mude "Detecção de carimbo" e/ou
"Classificação de tipo" para `modelo_treinado` (apontando o `.pt`
gerado), e/ou "Motor de OCR" para `paddleocr`.

Se um modelo/motor não puder ser carregado por qualquer motivo
(arquivo ausente, dependência faltando, CPU incompatível), o sistema
registra o erro no log e volta automaticamente para a estratégia
padrão (heurística/regras/Tesseract) — o processamento em lote nunca
é interrompido por causa disso.

Quando "Detecção de carimbo" está em `modelo_treinado`, o CAMP Vision
também tenta uma **2ª estratégia** por prancha sempre que o modelo
não encontra nada (ou só acha abaixo do limiar de confiança
configurado): a busca geométrica por regiões candidatas, verificada
por conteúdo (palavras-chave típicas de carimbo via OCR) — a mesma
usada no modo `heuristico`. Isso cobre casos em que o modelo erra mas
a posição/geometria típica do carimbo ainda dá pistas suficientes,
sem custar uma inferência extra por orientação testada (só entra em
jogo uma vez, já na orientação escolhida). Ver
`scanner/pipeline.py: _tentar_fallback_geometrico`.

Ao final de cada lote, além do CSV/XLSX/JSON, o CAMP Vision escreve
`relatorio.txt` na pasta de saída com um resumo (pranchas
processadas, quantas tiveram carimbo/texto lido, número de folha e
arquiteto identificados, projetos distintos, registros com erro) —
útil para comparar uma execução (ou uma versão do modelo/heurística
de detecção) com a próxima sem vasculhar o log inteiro.

### Formatos de entrada — JPG como principal

O fluxo recomendado é digitalizar em TIFF (guardado à parte, fora do
CAMP Vision, como master de preservação) e JPG (o que o CAMP Vision
de fato processa) — JPG é bem mais rápido de ler (sem a dependência
`imagecodecs`/LZW que TIFFs comprimidos exigem) e os arquivos são bem
menores, o que ajuda tanto a velocidade quanto o espaço em disco
usado durante o processamento (ver checagem de espaço em disco,
acima). O restante do pipeline (detecção de carimbo, OCR, correção de
grafia) é agnóstico a formato — trabalha sempre com arrays BGR depois
da leitura — mas duas coisas mudam de fato com JPG:

- **Cor**: scans coloridos passam pelo pipeline normalmente (a
  detecção converte pra escala de cinza internamente onde precisa,
  ex. `ocr/tesseract_ocr.py: _preparar_para_ocr`) — não é preciso
  configurar nada.
- **Resolução**: JPG raramente tem a tag de DPI gravada (ao contrário
  de TIFF, onde quase sempre está presente) — sem ela, o aviso de
  "resolução baixa" não tem como calcular DPI, então o CAMP Vision
  cai num piso de pixels absolutos (`RESOLUCAO_MINIMA_PIXELS_SEM_DPI`,
  `scanner/leitor_imagem.py`) como rede de segurança best-effort —
  não é tão preciso quanto o DPI real, mas pega o caso óbvio de um
  scan pequeno demais.

O arquivamento (`shutil.copy2`) e a exportação de metadados (exiftool)
não recodificam a imagem — o JPG original é copiado bit a bit, sem
perda adicional de qualidade em nenhuma etapa do processo.

### Pré-processamento de imagem antes do OCR

Antes de qualquer chamada ao Tesseract, o CAMP Vision reduz ruído de
granulação — mais comum em TIFFs digitalizados de acervo, e ainda mais
acentuado em pranchas **unidas** (duas metades escaneadas
separadamente e depois juntadas numa imagem só, às vezes com
exposição/contraste levemente diferentes entre as metades). A ordem
importa (testada empiricamente com ruído sintético forte): mediana
3x3 (remove a maior parte do ruído tipo sal-e-pimenta — sozinha já
resolve boa parte do problema), depois `fastNlMeansDenoising` (limpa
ruído gaussiano residual sem borrar as bordas das letras), depois
binarização por Otsu (limiar calculado automaticamente pela própria
imagem). Limiar ADAPTATIVO foi testado e descartado: amplifica ruído
residual em manchas com cara de texto, piorando o resultado em vez de
melhorar. Ver `ocr/tesseract_ocr.py: _preparar_para_ocr`.

### Carimbo dividido em duas caixas

Alguns acervos usam um carimbo de título dividido em DUAS caixas lado
a lado — ex.: bloco com nome/endereço do escritório à esquerda e uma
tabela institucional (cliente, projeto, escala, revisões) à direita,
cada uma com sua própria borda (achado com um TIFF real de um acervo
Eletropaulo/CARMONA). Sem tratamento especial, a busca geométrica
capturava só a caixa de maior pontuação, cortando a outra fora —
nesse caso, cortava justamente a tabela com projeto/escala. Agora,
depois de achar a melhor caixa, outras caixas na MESMA faixa vertical
e horizontalmente PRÓXIMAS a ela são fundidas numa única região antes
do recorte. Ver `scanner/detector_carimbo.py: _melhor_candidato_por_regiao`.

### Candidato único também precisa ser verificado

Quando a busca geométrica encontra candidato em SÓ UMA região (o caso
mais comum, já que a maioria das regiões buscadas não passa nem do
filtro geométrico básico), a função pulava a verificação por OCR e o
limiar mínimo inteiramente — devolvia o candidato direto, sem checar
se ele de fato tinha cara de carimbo, e sem logar nada (achado
investigando um arquivo real onde só 1 de 4 tentativas de orientação
aparecia no log de verdade — as outras 3 tinham encontrado candidato
único e retornado em silêncio, sem verificação nenhuma). Agora
candidato único passa pela MESMA verificação de conteúdo que
candidatos múltiplos — mais lento nesse caso específico (perde o
atalho que pulava o OCR), mas consistente: nenhum candidato geométrico
é aceito sem checar se o texto reconhecido de fato parece um carimbo.

### Orientação quando o modelo treinado falha em tudo

Quando o modelo treinado não encontra o carimbo em NENHUMA das 8
orientações testadas (comum quando o modelo tem desempenho ruim nesse
acervo específico — não é bug, é o modelo real ficando bem abaixo da
validação), a 2ª estratégia geométrica vira a única chance de achar o
carimbo — mas ela também depende da orientação estar certa.
Confiar só num heurístico de OCR genérico (não específico de carimbo)
para escolher a orientação fazia a 2ª estratégia buscar nos cantos
errados quando esse palpite errava a rotação (achado num caso real:
0.87 de confiança na orientação certa, só 0.36 — abaixo do limiar —
na orientação escolhida pelo heurístico genérico). A pontuação
puramente geométrica (sem OCR) também não serve para decidir a
orientação: testada na prática, sai praticamente idêntica nas 4
rotações. Por isso, nesse cenário específico, a 2ª estratégia completa
(com verificação por OCR, resolução real) é testada nas 4 rotações
básicas, com saída antecipada assim que uma cruza
`CONFIANCA_BOA_O_BASTANTE` — mais lento nesse pior caso (só quando o
modelo falhou por completo), mas encontra o carimbo onde antes
simplesmente desistia. Ver
`scanner/pipeline.py: _detectar_em_qualquer_orientacao`.

### Rastreando qual arquivo gerou cada linha do log

O processamento em lote roda vários arquivos ao mesmo tempo
(`ThreadPoolExecutor`), então mensagens intermediárias de detecção de
carimbo de arquivos diferentes ficam intercaladas no log por ordem de
chegada — sem indicação de qual arquivo cada uma pertence, era
impossível saber, por exemplo, quantas tentativas de orientação
realmente rodaram para um arquivo específico (precisou ser investigado
na prática, comparando tempos entre mensagens). Agora toda linha de
log emitida durante o processamento de um arquivo (em QUALQUER módulo,
inclusive os de detecção/OCR) ganha automaticamente o prefixo
`[nome_do_arquivo]`, correto mesmo com várias threads processando
arquivos diferentes ao mesmo tempo — via `contextvars`, por thread,
sem precisar passar o nome do arquivo por parâmetro em cada chamada de
log. Ver `utils/logger.py: arquivo_em_processamento`.

---

## Renomeação com código único e organização em pastas

Cada TIFF original é renomeado automaticamente com um código único de
arquivamento, seguindo o padrão:

```
{codigo_projeto}-{prancha}-{sequencial}.ext

Exemplo (projeto "Sami Bussab", prancha "P001.1"):
SB-P001.1-00001.tif
```

- **`codigo_projeto`**: sigla curta do projeto (ex.: "SB"), digitada
  no campo "Código do projeto" da tela principal — sugerida
  automaticamente a partir do nome da pasta selecionada.
- **`prancha`**: identificador da prancha extraído do carimbo.
- **`sequencial`**: contador único e persistente (não reinicia entre
  execuções), com zero-padding configurável.

O padrão e o número de dígitos são configuráveis em **Configurações**.

### Organização em pastas (ano/projeto)

Depois de renomeado, o arquivo é movido para uma estrutura de pastas
por ano e projeto — aplicação prática dos princípios de proveniência
e organicidade tratados na Resolução CONARQ/MGI nº 56/2024
("Diretrizes para o tratamento técnico de arquivos relacionados à
arquitetura e ao ambiente construído"):

```
<pasta selecionada>/<ano>/<nome do projeto>/SB-P001.1-00001.tif
```

Onde `<nome do projeto>` é, por padrão, o nome da própria pasta
selecionada (ex.: "Sami Bussab") e `<ano>` vem do ano extraído do
carimbo (ou "Ano desconhecido" se não identificado). O padrão de
pastas e a pasta raiz de destino são configuráveis em
**Configurações**; essa etapa pode ser desabilitada.

O `<ano>` usado na PASTA é um único valor por projeto — o ano mais
comum entre as pranchas daquele projeto — mesmo que pranchas
individuais do mesmo projeto tragam anos diferentes no carimbo (ou
nenhum ano legível). Isso evita que um único projeto fique
fragmentado em várias pastas de ano diferentes; o ano de cada
prancha continua sendo o lido individualmente no CSV e no EXIF.

### Metadados gravados no arquivo (EXIF/IPTC/XMP)

Além de renomear e organizar, o CAMP Vision grava os metadados
extraídos diretamente no próprio arquivo (descrição, autor/
arquiteto, cliente, palavras-chave com tipo/cidade/ano/fase etc.),
usando o [ExifTool](https://exiftool.org/). Isso torna o acervo
pesquisável por nome de projeto, arquiteto, cidade ou tipo de prancha
diretamente pelo Spotlight do macOS, Adobe Bridge, ou qualquer outro
leitor de metadados.

O campo **Copyright/Rights** combina o arquiteto com uma atribuição
institucional configurável (padrão: "CAMP - Casa da Arquitetura
Moderna Paulista"), no formato `"{arquiteto} / {atribuição}"` — não é
mais o nome do cliente (que é o dono/contratante da obra, não quem
tem os direitos sobre o desenho). Ajustável em **Configurações → 
Metadados no arquivo → Atribuição no Copyright** (em branco grava só
o arquiteto). Ver `utils/metadados_exif.py: _montar_copyright`.

Se o `exiftool` não estiver instalado, essa etapa é simplesmente
pulada (com aviso no log) — não interrompe o processamento. A opção
pode ser desabilitada na tela de Configurações.

### Banco de conhecimento (correção e aprendizado entre execuções)

O CAMP Vision guarda, em `~/.campvision/campvision.sqlite3`, um banco
de valores já vistos — arquiteto, cidade, **projeto**, **endereço de
obra** e **escala** — usado para corrigir erros de OCR por
similaridade textual (`database/repository.py`). Ele aprende
sozinho, lote após lote: toda vez que um projeto é lido junto com um
endereço, essa associação fica salva e passa a preencher o endereço
de OUTRAS pranchas do mesmo projeto que não trouxerem essa
informação no carimbo — inclusive em processamentos futuros, não só
dentro do lote atual. Uma associação já confirmada nunca é
sobrescrita por uma leitura isolada.

Um valor visto pela primeira vez fica em "quarentena": é lembrado
(para se reconhecer numa 2ª leitura parecida), mas só passa a corrigir
OUTRAS leituras diferentes dele depois de confirmado por uma leitura
seguinte — assim, um erro de OCR isolado, visto uma única vez, não
"vira vocabulário" e não vai puxar uma leitura correta pra perto de
si. Quando um valor é confirmado, a grafia mais completa entre as
leituras vistas (ex.: "Carlos Barjas Millan" em vez de "Carlos B
Millan") passa a ser a canônica usada nas próximas correções.

`tipo` (Planta, Corte, Fachada etc.) não passa por essa correção: é
uma classificação de categoria fixa (`classificacao/classificador.py`),
não texto livre de OCR, então corrigi-lo por similaridade abriria
espaço pra "inventar" categorias fora da lista fixa. `escritório`
também não tem um campo próprio hoje — o carimbo não distingue
explicitamente "arquiteto" de "escritório", então ambos caem no mesmo
campo `arquiteto`.

Esse banco também pode ser alimentado manualmente, sem reprocessar
nenhuma imagem, com `database/importar_conhecimento.py`:

```bash
# a partir de uma planilha de acervo já revisada à mão (colunas
# PROJETO, LOCAL, DATA, TIPO, NOME, ESCALA, FOLHA, OBSERVAÇÕES)
python -m database.importar_conhecimento --planilha acervo.xlsx

# a partir de um CSV de catalogação de um lote já processado — usa só
# os campos lidos diretamente do carimbo, ignorando o que a coluna
# Observações marca como propagado de outra prancha
python -m database.importar_conhecimento --catalogacao catalogacao.csv
```

---

## Roadmap (arquitetura já preparada para expansão)

- **Entrada em PDF**: `scanner/leitor_imagem.py` foi projetado com uma
  função `carregar_paginas` isolada; basta implementar um leitor
  equivalente para PDF mantendo a mesma interface (`ImagemCarregada`).
- **Reconhecimento de elementos gráficos adicionais** (escadas,
  vegetação, mobiliário, curvas de nível, cotas, indicação de norte,
  quadro de áreas, quadro de esquadrias, estruturas): o módulo
  `classificacao/` foi desenhado para receber novos classificadores
  especializados sem alterar o pipeline principal.
- **Novos motores de OCR/IA/detecção/classificação**: todos expostos
  por fábricas de estratégia (`ocr/motor.py`,
  `scanner/detector_carimbo.criar_detector`,
  `classificacao/classificador.criar_classificador`) com fallback
  automático — novas estratégias podem ser adicionadas sem alterar o
  restante do sistema.

---

## Licença

Projeto proprietário — uso interno definido pelo solicitante.
