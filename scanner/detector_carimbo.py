"""Detecção robusta de blocos de identificação/carimbos em pranchas arquitetônicas.

A estratégia deliberadamente separa localização e interpretação: primeiro encontramos
regiões promissoras nas bordas da prancha, depois usamos OCR para ranqueá-las. O
objetivo é recuperar carimbos históricos, inclusive quando não existe um retângulo
perfeito ou quando o bloco é uma faixa vertical/horizontal.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from scanner.leitor_imagem import reduzir_para_ocr
from utils.texto import normalizar_maiusculas as _normalizar_para_comparacao

logger = logging.getLogger("campvision.scanner.carimbo")

# Regiões deliberadamente generosas. O bloco de identificação nem sempre fica
# no canto e, nas pranchas do CAMP, pode ocupar uma faixa inteira da borda.
PROPORCAO_CANTO = 0.42
PROPORCAO_FAIXA = 0.24
PROPORCAO_FAIXA_ESTENDIDA = 0.34
AREA_MINIMA_RELATIVA = 0.002
AREA_MAXIMA_RELATIVA = 0.995
PONTUACAO_MINIMA = 0.12
LIMIAR_MINIMO_CONTEUDO_VERIFICADO = 0.28
LIMIAR_CANDIDATO_SEGUNDA_ANALISE = 0.22
TAMANHO_MAX_BUSCA_CONTORNO = 2400
TAMANHO_MAX_VERIFICACAO = 1800
MAX_CANDIDATOS_POR_REGIAO = 4

PALAVRAS_CHAVE_CARIMBO = (
    "ESCALA", "PROJETO", "ARQUITETO", "ARQUITETURA", "CLIENTE", "DATA",
    "PRANCHA", "PROPRIETARIO", "ENDERECO", "CIDADE", "FASE", "DESENHO",
    "DESENHOU", "RESPONSAVEL", "CREA", "CAU", "REV", "REVISAO", "FOLHA",
    "OBRA", "LOCAL", "MUNICIPIO", "ESTADO", "APROVADO", "VERIFICADO",
    "RUA", "AV", "AVENIDA", "ALAMEDA", "TELEFONE", "TEL", "FONE",
    "LTDA", "EIRELI", "ENG", "ARQ", "CONSTR", "EMPR", "IMOB",
    "URBANISMO", "ENGENHARIA", "RESIDENCIA", "IGREJA", "PLANTA", "CORTE",
    "DETALHE", "FACHADA", "PAVIMENTO", "PROJETISTA", "DESENHO", "REFERENCIA",
    "TABELA", "APROVACAO", "ASSINATURA", "NOME", "END.",
)

_PADRAO_TELEFONE = re.compile(r"\b(?:\(?\d{2}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}\b")
_PADRAO_ESCALA = re.compile(r"\b(?:1\s*[:/]\s*\d{1,5}|\d{1,5}\s*[:/]\s*1)\b")
_PADRAO_NUMERO = re.compile(r"\b(?:N[º°.]?|FOLHA|PRANCHA|DESENHO)\s*[:.-]?\s*[A-Z]?\d{1,4}\b")

REGIOES_VALIDAS = (
    "automatico", "superior_esquerdo", "superior_direito",
    "inferior_esquerdo", "inferior_direito", "faixa_direita",
    "faixa_inferior", "faixa_esquerda", "faixa_superior",
)

@dataclass
class CarimboDetectado:
    x: int
    y: int
    largura: int
    altura: int
    confianca: float
    canto: str

    def recortar(self, imagem: np.ndarray) -> np.ndarray:
        h, w = imagem.shape[:2]
        x0 = max(0, min(self.x, w - 1))
        y0 = max(0, min(self.y, h - 1))
        x1 = max(x0 + 1, min(self.x + self.largura, w))
        y1 = max(y0 + 1, min(self.y + self.altura, h))
        return imagem[y0:y1, x0:x1].copy()


def _regioes_candidatas(altura_img: int, largura_img: int, regiao_fixa: str = "automatico"):
    rh = int(altura_img * PROPORCAO_CANTO)
    rw = int(largura_img * PROPORCAO_CANTO)
    fh = int(altura_img * PROPORCAO_FAIXA)
    fw = int(largura_img * PROPORCAO_FAIXA)
    feh = int(altura_img * PROPORCAO_FAIXA_ESTENDIDA)
    few = int(largura_img * PROPORCAO_FAIXA_ESTENDIDA)
    todas = {
        "superior_esquerdo": (0, 0, rw, rh),
        "superior_direito": (largura_img-rw, 0, largura_img, rh),
        "inferior_esquerdo": (0, altura_img-rh, rw, altura_img),
        "inferior_direito": (largura_img-rw, altura_img-rh, largura_img, altura_img),
        "faixa_direita": (largura_img-fw, 0, largura_img, altura_img),
        "faixa_inferior": (0, altura_img-fh, largura_img, altura_img),
        "faixa_esquerda": (0, 0, fw, altura_img),
        "faixa_superior": (0, 0, largura_img, fh),
        # Faixas estendidas ajudam quando o bloco fica um pouco para dentro.
        "faixa_direita_estendida": (largura_img-few, 0, largura_img, altura_img),
        "faixa_esquerda_estendida": (0, 0, few, altura_img),
        "faixa_superior_estendida": (0, 0, largura_img, feh),
        "faixa_inferior_estendida": (0, altura_img-feh, largura_img, altura_img),
    }
    if regiao_fixa == "automatico":
        return todas
    if regiao_fixa in todas:
        return {regiao_fixa: todas[regiao_fixa]}
    logger.warning("Região de carimbo '%s' desconhecida — usando busca automática.", regiao_fixa)
    return todas


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0, y0 = max(ax,bx), max(ay,by)
    x1, y1 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    inter = max(0,x1-x0)*max(0,y1-y0)
    den = aw*ah + bw*bh - inter
    return inter/den if den else 0.0


def _pontuar_contorno(contorno, area_regiao, regiao_recortada):
    x, y, w, h = cv2.boundingRect(contorno)
    area = w*h
    if not area:
        return 0.0
    rel = area/area_regiao
    if not AREA_MINIMA_RELATIVA <= rel <= AREA_MAXIMA_RELATIVA:
        return 0.0
    solidez = cv2.contourArea(contorno)/area
    per = cv2.arcLength(contorno, True)
    aprox = cv2.approxPolyDP(contorno, 0.025*per, True) if per else []
    ret = solidez if len(aprox) in (4,5,6,7,8) else solidez*0.65
    proporcao = w/h if h else 0
    prop = 1.0 if 0.12 <= proporcao <= 12 else 0.55
    rec = regiao_recortada[y:y+h, x:x+w]
    bord = cv2.Canny(rec, 40, 130) if rec.size else np.empty((0,0), np.uint8)
    dens = min(float(np.count_nonzero(bord))/bord.size*8, 1.0) if bord.size else 0.0
    # Para não excluir blocos sem moldura, a geometria é só uma evidência.
    return 0.30*ret + 0.25*min(rel/0.70,1.0) + 0.15*prop + 0.30*dens


def _candidatos_por_regiao(imagem, regiao_fixa="automatico"):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
    out = {}
    for nome, (x0,y0,x1,y1) in _regioes_candidatas(*gray.shape[:2], regiao_fixa).items():
        reg = gray[y0:y1,x0:x1]
        if reg.size == 0: continue
        # Duas morfologias: uma preserva caixas; outra une linhas de tabela/texto.
        variantes = []
        for block in ((3,3),(9,3),(3,9),(15,5),(5,15)):
            th = cv2.adaptiveThreshold(reg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, block)
            variantes.append(cv2.morphologyEx(th, cv2.MORPH_CLOSE, k))
        caixas=[]
        area_reg=reg.shape[0]*reg.shape[1]
        for binaria in variantes:
            contornos,_=cv2.findContours(binaria,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for c in contornos:
                p=_pontuar_contorno(c,area_reg,reg)
                if p >= PONTUACAO_MINIMA:
                    caixas.append((p,cv2.boundingRect(c)))
        caixas.sort(key=lambda z:z[0],reverse=True)
        selecionadas=[]
        for p,box in caixas:
            if any(_iou(box,b)>=0.65 for _,b in selecionadas): continue
            x,y,w,h=box
            # Evita candidatos que sejam só um risco/linha extremamente fino.
            if min(w,h) < 12: continue
            selecionadas.append((p,box))
            if len(selecionadas)>=MAX_CANDIDATOS_POR_REGIAO: break
        if selecionadas:
            # O maior bloco pode ser dividido em duas caixas. Une caixas próximas.
            p0,(cx,cy,cw,ch)=selecionadas[0]
            cx1,cy1=cx+cw,cy+ch
            for p,(x,y,w,h) in selecionadas[1:]:
                ov=min(cy1,y+h)-max(cy,y)
                hm=min(cy1-cy,h)
                gap=max(0,max(x,cx)-min(x+w,cx1))
                if hm>0 and ov/hm>=0.55 and gap<=max(20,int(0.12*(cx1-cx))):
                    cx,cy=min(cx,x),min(cy,y); cx1,cy1=max(cx1,x+w),max(cy1,y+h)
            out[nome]=CarimboDetectado(x0+cx,y0+cy,cx1-cx,cy1-cy,p0,nome)
    return out


def _palavra_corresponde_a_chave(palavra, chave):
    if len(chave)<3: return palavra==chave
    maior,menor=max(len(palavra),len(chave)),min(len(palavra),len(chave))
    if menor/maior < 0.68: return False
    return difflib.SequenceMatcher(None,palavra,chave).ratio() >= 0.68


def _contar_acertos_aproximados(texto_normalizado):
    acertos=0
    for palavra in texto_normalizado.split():
        p=re.sub(r"[^A-Z0-9º]","",palavra)
        if len(p)<3: continue
        if any(_palavra_corresponde_a_chave(p,k) for k in PALAVRAS_CHAVE_CARIMBO): acertos+=1
    return acertos


def _pontuar_conteudo(texto):
    texto_normalizado=_normalizar_para_comparacao(texto or "")
    acertos=_contar_acertos_aproximados(texto_normalizado)
    palavras=len((texto or "").split())
    campos=sum(bool(p) for p in (_PADRAO_TELEFONE.search(texto or ""), _PADRAO_ESCALA.search(texto or ""), _PADRAO_NUMERO.search(texto or "")))
    # Um bloco de identificação pode ter palavras que não estão na lista;
    # portanto densidade e padrões entram como evidência secundária.
    chave=min(acertos/4,1.0)
    dens=min(palavras/28,1.0)
    campo=min(campos/2,1.0)
    return 0.62*chave + 0.23*campo + 0.15*dens


def _reduzir_para_verificacao(recorte):
    return reduzir_para_ocr(recorte,TAMANHO_MAX_VERIFICACAO)


def _gerar_candidatos_com_fallback(imagem, regiao_fixa):
    candidatos=_candidatos_por_regiao(imagem,regiao_fixa)
    # Se a heurística não achou caixas, ainda oferecemos as próprias regiões de
    # borda como candidatos. Isso é crucial para carimbos sem moldura fechada.
    if not candidatos:
        h,w=imagem.shape[:2]
        for nome,(x0,y0,x1,y1) in _regioes_candidatas(h,w,regiao_fixa).items():
            candidatos[nome]=CarimboDetectado(x0,y0,x1-x0,y1-y0,0.10,nome)
    return candidatos


def detectar_carimbo(imagem, regiao_fixa="automatico"):
    candidatos=_gerar_candidatos_com_fallback(imagem,regiao_fixa)
    if not candidatos:
        logger.warning("Carimbo/bloco de identificação não localizado (região='%s')",regiao_fixa)
        return None
    melhor=max(candidatos.values(),key=lambda c:c.confianca)
    return melhor


def detectar_carimbo_verificado(imagem, ocr_fn: Callable[[np.ndarray,list],object], idiomas_ocr=None,
                                regiao_fixa="automatico", limiar_minimo=LIMIAR_MINIMO_CONTEUDO_VERIFICADO):
    h,w=imagem.shape[:2]
    escala=min(1.0,TAMANHO_MAX_BUSCA_CONTORNO/max(h,w))
    busca=cv2.resize(imagem,None,fx=escala,fy=escala,interpolation=cv2.INTER_AREA) if escala<1 else imagem
    candidatos=_gerar_candidatos_com_fallback(busca,regiao_fixa)
    if escala<1:
        f=1/escala
        candidatos={n:CarimboDetectado(int(c.x*f),int(c.y*f),int(c.largura*f),int(c.altura*f),c.confianca,c.canto) for n,c in candidatos.items()}
    avaliados=[]
    for nome,c in candidatos.items():
        rec=c.recortar(imagem)
        try:
            r=ocr_fn(_reduzir_para_verificacao(rec),idiomas_ocr or ["pt"])
            texto=getattr(r,"texto","") or ""
            conf_ocr=float(getattr(r,"confianca_media",0.0) or 0.0)
        except Exception as exc:
            logger.debug("Falha OCR no candidato '%s': %s",nome,exc)
            texto=""; conf_ocr=0.0
        conteudo=_pontuar_conteudo(texto)
        # Conteúdo domina a decisão; geometria apenas desempata. Assim um
        # carimbo sem moldura pode vencer uma legenda perfeitamente retangular.
        final=0.20*c.confianca + 0.68*conteudo + 0.12*min(conf_ocr/100 if conf_ocr>1 else conf_ocr,1.0)
        avaliados.append((final,c,texto,conteudo))
        logger.debug("Candidato '%s': geometria=%.2f conteúdo=%.2f OCR=%.2f final=%.2f",nome,c.confianca,conteudo,conf_ocr,final)
    if not avaliados: return None
    avaliados.sort(key=lambda z:z[0],reverse=True)
    final,c,texto,conteudo=avaliados[0]
    # Segunda análise: candidatos medianos recebem um OCR em recorte ampliado,
    # preservando margem ao redor do bloco. Isso recupera textos parcialmente
    # cortados pela caixa geométrica.
    if final < limiar_minimo and final >= LIMIAR_CANDIDATO_SEGUNDA_ANALISE:
        margem_x=max(20,int(c.largura*0.15)); margem_y=max(20,int(c.altura*0.15))
        x0=max(0,c.x-margem_x); y0=max(0,c.y-margem_y)
        x1=min(w,c.x+c.largura+margem_x); y1=min(h,c.y+c.altura+margem_y)
        rec=imagem[y0:y1,x0:x1]
        try:
            r=ocr_fn(_reduzir_para_verificacao(rec),idiomas_ocr or ["pt"])
            t2=getattr(r,"texto","") or ""
            p2=_pontuar_conteudo(t2)
            f2=0.10*c.confianca+0.78*p2+0.12*min(float(getattr(r,"confianca_media",0) or 0)/100,1)
            if f2>final:
                final=f2; texto=t2; c=CarimboDetectado(x0,y0,x1-x0,y1-y0,final,c.canto)
        except Exception: pass
    if final < limiar_minimo:
        logger.info("Nenhum bloco de identificação passou o limiar: melhor='%s' confiança %.2f (limiar %.2f)",c.canto,final,limiar_minimo)
        return None
    c.confianca=final
    logger.info("Bloco de identificação escolhido: região '%s', confiança final %.2f",c.canto,final)
    return c


def criar_detector(modo="heuristico", caminho_modelo=None, confianca_minima=0.5,
                   regiao_fixa="automatico", ocr_fn=None, idiomas_ocr=None, tamanho_imagem_ml=0):
    if modo=="modelo_treinado" and caminho_modelo:
        try:
            from ml.detector_carimbo_ml import DetectorCarimboML
            modelo=DetectorCarimboML(caminho_modelo,confianca_minima=confianca_minima,
                                     tamanho_imagem=tamanho_imagem_ml if tamanho_imagem_ml>0 else None)
            logger.info("Detector de carimbo por modelo treinado carregado: %s",caminho_modelo)
            return modelo.detectar
        except Exception as exc:
            logger.error("Falha ao carregar modelo treinado (%s). Usando heurística.",exc)
    if ocr_fn is not None:
        return lambda imagem: detectar_carimbo_verificado(imagem,ocr_fn,idiomas_ocr,regiao_fixa=regiao_fixa)
    return lambda imagem: detectar_carimbo(imagem,regiao_fixa=regiao_fixa)


def corrigir_orientacao_do_carimbo(recorte, ocr_fn, idiomas):
    from scanner.leitor_imagem import _gerar_transformacoes, _rotacionar
    reduzido=reduzir_para_ocr(recorte,1400)
    melhor=((0,False),(-1,-1.0))
    for chave,candidata in _gerar_transformacoes(reduzido).items():
        try: r=ocr_fn(candidata,idiomas)
        except Exception: continue
        score=(_contar_acertos_aproximados(_normalizar_para_comparacao(getattr(r,"texto","") or "")),float(getattr(r,"confianca_media",0) or 0))
        if score>melhor[1]: melhor=(chave,score)
    (angulo,espelhado),score=melhor
    if not angulo and not espelhado: return recorte
    base=cv2.flip(recorte,1) if espelhado else recorte
    logger.info("Carimbo endireitado (rotação=%d°, espelhado=%s; %d palavra(s) de identificação reconhecida(s)).",angulo,espelhado,score[0])
    return _rotacionar(base,angulo)
