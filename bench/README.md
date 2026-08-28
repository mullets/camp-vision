# Avaliador do CAMP Vision

Mede objetivamente a qualidade da catalogação contra um gabarito
feito à mão. Existe para substituir o "achismo": em vez de olhar meia
dúzia de recortes e tentar adivinhar se uma mudança melhorou, roda-se
o avaliador antes e depois e compara-se o placar.

## Por que isso importa

Sem medição, todo ajuste é uma aposta — e ajustes que melhoram um
acervo frequentemente pioram outro. Com o placar, cada mudança tem
uma resposta objetiva, e dá para acompanhar a meta (localização de
carimbo acima de 95-98%, metadados principais acima de 90%) em vez de
estimá-la.

## Preparando o conjunto de referência

1. Escolha **pranchas inteiras** (não recortes de carimbo) — de
   preferência 15 a 30, misturando acervos e escritórios diferentes,
   incluindo casos difíceis (carimbo girado, cópia degradada, prancha
   sem carimbo).
2. Coloque todas numa pasta.
3. Preencha `gabarito.csv` com o que você, olhando a prancha, sabe
   que é o certo.

Só as colunas preenchidas são cobradas — deixe em branco o que não
quiser avaliar naquela linha.

```csv
arquivo,carimbo,folha,tipo,escala,ano,arquiteto,projeto,endereco
DEST2696.tif,sim,1,Planta,1:250,1974,Sami Bussab,SABESP,RUA PARAMU
DEST2695.tif,nao,,,,,,,
```

- `carimbo`: `sim` se a prancha tem carimbo legível, `nao` se não tem
  (serve para medir a taxa de localização sem punir pranchas que
  genuinamente não têm carimbo).
- `folha`, `tipo`, `escala`, `ano`: comparados de forma exata.
- `arquiteto`, `projeto`, `endereco`: basta o trecho esperado
  aparecer no resultado (o OCR costuma trazer algo a mais).

## Rodando

Modo heurístico (sem modelo treinado):

```bash
python -m bench.avaliar --pasta ~/acervo_referencia --gabarito bench/gabarito.csv
```

Com o modelo treinado:

```bash
python -m bench.avaliar --pasta ~/acervo_referencia \
                        --gabarito bench/gabarito.csv \
                        --modelo ~/Downloads/modelos/treino_carimbo/weights/best.pt \
                        --confianca 0.3
```

O avaliador **nunca renomeia, move ou altera** os arquivos avaliados:
renomeação, arquivamento e gravação de EXIF são desligados à força, e
a saída (recortes de carimbo e texto de OCR de cada prancha) vai para
uma pasta temporária, cujo caminho aparece no fim da execução.

## Lendo o placar

```
carimbo localizado   87.5%  (14/16)
folha                60.0%  (9/15)
tipo                 73.3%  (11/15)
```

Os "exemplos de erro" no fim mostram, por campo, o que era esperado e
o que saiu — é por onde começar a investigar.

Guarde o placar de cada rodada. A pergunta que importa não é "está
bom?", é "melhorou em relação à rodada anterior, e em qual campo?".
