"""
database/repository.py
=======================
Camada de acesso a dados: operações de leitura/escrita sobre o banco
de conhecimento e o histórico de pranchas catalogadas.

Também implementa a "correção inteligente" de nomes (arquitetos,
escritórios, cidades) usando similaridade textual, de forma que
variações de OCR como:

    "Carlos B Millan", "Carlos Barjas Mlllan", "Carlos Millan"

sejam normalizadas para a forma canônica já conhecida pelo banco,
por exemplo "Carlos Barjas Millan".

A similaridade usa `difflib` (biblioteca padrão do Python) em vez de
uma dependência externa como RapidFuzz — não precisa de nenhum
pacote compilado, o que garante que funcione em qualquer Mac (mesmo
hardware antigo sem AVX/AVX2) sem depender de wheels pré-compiladas
disponíveis para aquela combinação exata de macOS/Python.
"""

from __future__ import annotations

import difflib
import logging
import unicodedata
from typing import Optional

from sqlalchemy.orm import Session

from database.models import (
    Arquiteto, Escritorio, Cidade, Tipo, Escala, Projeto, Endereco, Prancha, SequenciaCodigo,
)

logger = logging.getLogger("campvision.database")

LIMIAR_SIMILARIDADE = 0.85  # 0-1 (difflib), quanto maior, mais rigoroso
LIMIAR_CONFIANCA = 2  # nº de vezes que um valor precisa ser visto antes de virar alvo de correção pra OUTRAS leituras


def normalizar_texto(texto: str) -> str:
    """Remove acentos, espaços duplicados e normaliza caixa para comparação."""
    texto = texto.strip()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = " ".join(texto.split())
    return texto.upper()


def _melhor_correspondencia(alvo: str, candidatos: list[str]) -> Optional[tuple[str, float]]:
    """Encontra, entre os candidatos normalizados, o mais parecido com
    o alvo (também já normalizado), usando SequenceMatcher do
    difflib. Compara tanto a string inteira quanto suas palavras
    reordenadas (como o RapidFuzz's token_sort_ratio fazia), para
    tolerar diferenças de ordem entre nome/sobrenome."""
    if not candidatos:
        return None

    alvo_ordenado = " ".join(sorted(alvo.split()))
    melhor_candidato: Optional[str] = None
    melhor_pontuacao = 0.0

    for candidato in candidatos:
        candidato_ordenado = " ".join(sorted(candidato.split()))
        pontuacao_direta = difflib.SequenceMatcher(None, alvo, candidato).ratio()
        pontuacao_ordenada = difflib.SequenceMatcher(None, alvo_ordenado, candidato_ordenado).ratio()
        pontuacao = max(pontuacao_direta, pontuacao_ordenada)

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_candidato = candidato

    return (melhor_candidato, melhor_pontuacao) if melhor_candidato is not None else None


class ConhecimentoRepository:
    """Repositório de entidades de conhecimento reutilizáveis entre execuções."""

    def __init__(self, session: Session):
        self.session = session

    # ---------------------------------------------------------------
    # Arquitetos
    # ---------------------------------------------------------------
    def sugerir_arquiteto(self, nome_ocr: Optional[str]) -> Optional[str]:
        """Dado um nome vindo do OCR/IA, retorna o nome canônico mais
        provável já cadastrado no banco, ou o próprio nome se nenhum
        candidato suficientemente similar for encontrado.

        Um nome visto pela primeira vez fica em "quarentena": ele é
        lembrado (pra se reconhecer numa 2ª leitura parecida), mas
        ainda não é usado para corrigir leituras DIFERENTES dele — só
        depois de confirmado por uma leitura seguinte (idêntica ou
        parecida) é que passa a valer como vocabulário confiável. Isso
        evita que um erro de OCR isolado, visto uma única vez, "puxe"
        uma leitura correta e diferente na próxima prancha."""
        if not nome_ocr:
            return nome_ocr

        candidatos = self.session.query(Arquiteto).all()
        if not candidatos:
            self._cadastrar_arquiteto(nome_ocr)
            return nome_ocr

        alvo = normalizar_texto(nome_ocr)
        confiaveis = {normalizar_texto(a.nome): a for a in candidatos if a.contagem >= LIMIAR_CONFIANCA}
        quarentena = {normalizar_texto(a.nome): a for a in candidatos if a.contagem < LIMIAR_CONFIANCA}

        melhor = _melhor_correspondencia(alvo, list(confiaveis.keys()))
        if melhor and melhor[1] >= LIMIAR_SIMILARIDADE:
            registro = confiaveis[melhor[0]]
            nome_canonico = self._reforcar(registro, "nome", nome_ocr)
            if nome_canonico != nome_ocr:
                logger.info("Correção inteligente: '%s' -> '%s' (score=%.2f)",
                            nome_ocr, nome_canonico, melhor[1])
            return nome_canonico

        melhor_quarentena = _melhor_correspondencia(alvo, list(quarentena.keys()))
        if melhor_quarentena and melhor_quarentena[1] >= LIMIAR_SIMILARIDADE:
            registro = quarentena[melhor_quarentena[0]]
            nome_canonico = self._reforcar(registro, "nome", nome_ocr)
            if nome_canonico != nome_ocr:
                logger.info("Correção inteligente (confirmando 2ª leitura): '%s' -> '%s' (score=%.2f)",
                            nome_ocr, nome_canonico, melhor_quarentena[1])
            return nome_canonico

        # Nenhum similar suficiente: cadastra como novo (em quarentena)
        self._cadastrar_arquiteto(nome_ocr)
        return nome_ocr

    def _cadastrar_arquiteto(self, nome: str) -> Arquiteto:
        existente = self.session.query(Arquiteto).filter_by(nome=nome).first()
        if existente:
            return existente
        novo = Arquiteto(nome=nome, nome_normalizado=normalizar_texto(nome), contagem=1)
        self.session.add(novo)
        self.session.commit()
        return novo

    # ---------------------------------------------------------------
    # Cidades / Escritórios / Tipos / Escalas (mesma lógica genérica)
    # ---------------------------------------------------------------
    def sugerir_generico(self, model, nome_ocr: Optional[str]) -> Optional[str]:
        """Mesma lógica de correção com quarentena de `sugerir_arquiteto`,
        genérica para os modelos que só têm um campo de texto (`nome`
        ou `valor`) — Cidade, Escritorio, Tipo, Escala, Projeto, Endereco."""
        if not nome_ocr:
            return nome_ocr
        campo = "nome" if hasattr(model, "nome") else "valor"
        candidatos = self.session.query(model).all()
        if not candidatos:
            self._cadastrar_generico(model, campo, nome_ocr)
            return nome_ocr

        alvo = normalizar_texto(nome_ocr)
        confiaveis = {normalizar_texto(getattr(c, campo)): c for c in candidatos if c.contagem >= LIMIAR_CONFIANCA}
        quarentena = {normalizar_texto(getattr(c, campo)): c for c in candidatos if c.contagem < LIMIAR_CONFIANCA}

        melhor = _melhor_correspondencia(alvo, list(confiaveis.keys()))
        if melhor and melhor[1] >= LIMIAR_SIMILARIDADE:
            return self._reforcar(confiaveis[melhor[0]], campo, nome_ocr)

        melhor_quarentena = _melhor_correspondencia(alvo, list(quarentena.keys()))
        if melhor_quarentena and melhor_quarentena[1] >= LIMIAR_SIMILARIDADE:
            return self._reforcar(quarentena[melhor_quarentena[0]], campo, nome_ocr)

        self._cadastrar_generico(model, campo, nome_ocr)
        return nome_ocr

    def _reforcar(self, registro, campo: str, valor_visto: str) -> str:
        """Confirma uma nova ocorrência de um valor já cadastrado:
        soma 1 à contagem (o que pode promovê-lo de "quarentena" para
        "confiável" — ver LIMIAR_CONFIANCA).

        A grafia canônica só é atualizada para uma variante mais
        completa ENQUANTO o valor ainda está em quarentena (ainda
        não confirmado) — ex.: "Julio R." confirmado por "Julio R.
        Katinsky" na 2ª leitura. Uma vez confiável, a grafia fica
        CONGELADA: nenhuma leitura seguinte, por mais "completa" que
        pareça, muda mais o nome canônico.

        Isso é essencial: sem congelar, um único erro de OCR que por
        acaso resulte numa string mais comprida (ex.: 'HOSWALDO
        CORREA GONÇALVES', com um H espúrio, 1 caractere mais longa
        que 'Oswaldo Correa Goncalves') sequestraria a grafia
        canônica de um nome já confirmado centenas de vezes — e
        passaria a "corrigir" todas as leituras seguintes, corretas,
        para a forma errada. Foi exatamente isso que aconteceu num
        lote real antes desse ajuste (ver histórico)."""
        ja_era_confiavel = registro.contagem >= LIMIAR_CONFIANCA
        registro.contagem += 1
        if not ja_era_confiavel:
            valor_atual = getattr(registro, campo)
            if valor_visto and len(valor_visto) > len(valor_atual):
                setattr(registro, campo, valor_visto)
        self.session.commit()
        return getattr(registro, campo)

    def _cadastrar_generico(self, model, campo: str, valor: str):
        existente = self.session.query(model).filter_by(**{campo: valor}).first()
        if existente:
            return existente
        kwargs = {campo: valor, "contagem": 1}
        if model is Arquiteto:
            kwargs["nome_normalizado"] = normalizar_texto(valor)
        novo = model(**kwargs)
        self.session.add(novo)
        self.session.commit()
        self.session.refresh(novo)
        return novo

    def sugerir_cidade(self, nome_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Cidade, nome_ocr)

    def sugerir_escritorio(self, nome_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Escritorio, nome_ocr)

    def sugerir_tipo(self, nome_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Tipo, nome_ocr)

    def sugerir_escala(self, nome_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Escala, nome_ocr)

    # ---------------------------------------------------------------
    # Projetos / Endereços de obra
    # ---------------------------------------------------------------
    def sugerir_projeto(self, nome_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Projeto, nome_ocr)

    def sugerir_endereco(self, endereco_ocr: Optional[str]) -> Optional[str]:
        return self.sugerir_generico(Endereco, endereco_ocr)

    def endereco_do_projeto(self, nome_projeto: Optional[str]) -> Optional[str]:
        """Se o projeto já tem um endereço de obra conhecido —
        cadastrado manualmente (ex.: importado de um índice de acervo
        já revisado) ou aprendido num processamento anterior —, devolve
        esse endereço. Serve para preencher o endereço de uma prancha
        cujo carimbo não trouxe essa informação, mas cujo projeto já
        foi identificado (por leitura direta ou por correção)."""
        if not nome_projeto:
            return None
        candidatos = (
            self.session.query(Projeto)
            .filter(Projeto.endereco_conhecido.isnot(None))
            .all()
        )
        if not candidatos:
            return None

        alvo = normalizar_texto(nome_projeto)
        nomes = {normalizar_texto(p.nome): p for p in candidatos}
        melhor = _melhor_correspondencia(alvo, list(nomes.keys()))
        if melhor and melhor[1] >= LIMIAR_SIMILARIDADE:
            return nomes[melhor[0]].endereco_conhecido
        return None

    def registrar_endereco_do_projeto(self, nome_projeto: Optional[str], endereco: Optional[str]) -> None:
        """Associa um endereço de obra a um projeto. Usado tanto na
        importação de dados curados (planilhas de acervo revisadas à
        mão) quanto, automaticamente, sempre que uma prancha traz
        projeto E endereço lidos juntos — é assim que o banco aprende
        sozinho, lote após lote, sem precisar de planilha nenhuma.

        NUNCA sobrescreve um endereço já conhecido para o projeto —
        mesma regra de segurança usada na propagação entre pranchas
        (scanner/propagacao.py): só completa o que estava vazio, para
        que uma leitura isolada e errada não apague uma associação já
        confirmada."""
        if not nome_projeto or not endereco:
            return
        projeto = self.session.query(Projeto).filter_by(nome=nome_projeto).first()
        if projeto is None:
            projeto = self._cadastrar_generico(Projeto, "nome", nome_projeto)
        if projeto and not projeto.endereco_conhecido:
            projeto.endereco_conhecido = endereco
            self.session.commit()


class PranchaRepository:
    """Repositório de operações CRUD sobre os registros de pranchas catalogadas."""

    def __init__(self, session: Session):
        self.session = session

    def salvar(self, prancha: Prancha) -> Prancha:
        self.session.add(prancha)
        self.session.commit()
        self.session.refresh(prancha)
        return prancha

    def listar_todas(self) -> list[Prancha]:
        return self.session.query(Prancha).order_by(Prancha.processado_em).all()

    def contar(self) -> int:
        return self.session.query(Prancha).count()


class SequenciaRepository:
    """Controla o último número sequencial usado por código de
    projeto, de forma persistente — o contador nunca reinicia entre
    execuções, garantindo que o código gerado na renomeação seja
    sempre único para aquele projeto."""

    def __init__(self, session: Session):
        self.session = session

    def proximo_numero(self, codigo_projeto: str) -> int:
        """Reserva e retorna o próximo número sequencial para o código
        de projeto informado, incrementando e persistindo o contador
        imediatamente (a chamada deve ser serializada externamente por
        um lock, pois SQLite não garante atomicidade entre múltiplas
        conexões concorrentes para este padrão leitura-then-escrita)."""
        registro = self.session.query(SequenciaCodigo).filter_by(codigo_projeto=codigo_projeto).first()
        if registro is None:
            registro = SequenciaCodigo(codigo_projeto=codigo_projeto, ultimo_numero=0)
            self.session.add(registro)

        registro.ultimo_numero += 1
        self.session.commit()
        self.session.refresh(registro)
        return registro.ultimo_numero
