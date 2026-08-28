"""
database/models.py
===================
Modelos ORM (SQLAlchemy) do banco de conhecimento do CAMP Vision.

O banco guarda tanto o "conhecimento" reutilizável entre execuções
(arquitetos, escritórios, cidades, tipos, escalas — usado para
correção inteligente e sugestões) quanto o histórico de pranchas
catalogadas.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)


class Base(DeclarativeBase):
    pass


class Arquiteto(Base):
    __tablename__ = "arquitetos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nome: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    nome_normalizado: Mapped[str] = mapped_column(String(255), index=True)
    # Quantas vezes este valor foi visto (1ª leitura, ou confirmado por
    # uma leitura seguinte parecida) — ver LIMIAR_CONFIANCA em
    # database/repository.py. Um valor visto uma única vez ainda não é
    # usado para corrigir OUTRAS leituras diferentes dele, só para
    # reconhecer a si mesmo se aparecer de novo — assim um erro de OCR
    # isolado não vira "vocabulário confiável" só por ter sido a
    # primeira leitura.
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Escritorio(Base):
    __tablename__ = "escritorios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nome: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Cidade(Base):
    __tablename__ = "cidades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nome: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    uf: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Tipo(Base):
    __tablename__ = "tipos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nome: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Escala(Base):
    __tablename__ = "escalas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    valor: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Projeto(Base):
    """Nome de projeto/obra já visto — usado para corrigir grafias de
    OCR por similaridade, do mesmo jeito que Arquiteto/Cidade/etc.

    `endereco_conhecido` guarda o endereço de obra já associado a este
    projeto (vindo de planilhas de acervo já curadas, como um índice
    físico revisado à mão, ou aprendido durante o próprio
    processamento quando projeto e endereço aparecem juntos numa
    prancha) — permite preencher o endereço de OUTRAS pranchas do
    mesmo projeto quando o carimbo delas não traz essa informação."""

    __tablename__ = "projetos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nome: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    endereco_conhecido: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class Endereco(Base):
    """Endereço de obra já visto — só para correção de grafia por
    similaridade (ver Projeto.endereco_conhecido para a associação
    endereço -> projeto)."""

    __tablename__ = "enderecos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    valor: Mapped[str] = mapped_column(String(500), unique=True, index=True)
    contagem: Mapped[int] = mapped_column(Integer, default=1)


class SequenciaCodigo(Base):
    """Controle do último número sequencial usado para cada código de
    projeto, garantindo que o código único gerado na renomeação nunca
    se repita — mesmo entre execuções diferentes do programa."""

    __tablename__ = "sequencias_codigo"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    codigo_projeto: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    ultimo_numero: Mapped[int] = mapped_column(Integer, default=0)


class Prancha(Base):
    """Registro de catalogação de uma prancha (um TIFF processado)."""

    __tablename__ = "pranchas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    arquivo: Mapped[str] = mapped_column(String(1024), index=True)
    arquivo_original: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    codigo_gerado: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    projeto: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cliente: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    arquiteto: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cidade: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    endereco: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ano: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    prancha_titulo: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    numero: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    escala: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tipo: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    fase: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    observacoes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    confianca_ocr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confianca_ia: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    caminho_miniatura: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    caminho_carimbo: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    erro: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


_TABELAS_COM_CONTAGEM = ("arquitetos", "escritorios", "cidades", "tipos", "escalas", "projetos", "enderecos")


def _migrar_colunas_novas(engine) -> None:
    """SQLAlchemy `create_all` só cria tabelas que ainda não existem —
    não adiciona colunas novas a tabelas já existentes de uma execução
    anterior do app. Sem isto, quem já tinha um banco de conhecimento
    de uma versão anterior (sem a coluna `contagem`) quebraria ao
    tentar ler/gravar essas tabelas. SQLite suporta `ALTER TABLE ...
    ADD COLUMN` com valor padrão, então aplicamos isso aqui, uma vez,
    de forma idempotente (checando antes se a coluna já existe)."""
    from sqlalchemy import inspect, text

    inspetor = inspect(engine)
    nomes_tabelas_existentes = set(inspetor.get_table_names())

    with engine.begin() as conexao:
        for tabela in _TABELAS_COM_CONTAGEM:
            if tabela not in nomes_tabelas_existentes:
                continue  # tabela nova, criada por create_all já com a coluna
            colunas = {c["name"] for c in inspetor.get_columns(tabela)}
            if "contagem" not in colunas:
                conexao.execute(text(f"ALTER TABLE {tabela} ADD COLUMN contagem INTEGER DEFAULT 1"))


def criar_engine(db_path: str):
    """Cria a engine SQLAlchemy e garante que as tabelas (e colunas
    novas em tabelas já existentes) existam."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    _migrar_colunas_novas(engine)
    return engine


def criar_sessao(db_path: str) -> Session:
    """Retorna uma nova sessão SQLAlchemy conectada ao banco em db_path."""
    engine = criar_engine(db_path)
    SessionLocal = sessionmaker(bind=engine, future=True)
    return SessionLocal()
