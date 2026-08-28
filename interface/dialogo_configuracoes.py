"""
interface/dialogo_configuracoes.py
====================================
Tela de configurações (Tkinter): idioma, pasta padrão, qualidade das
miniaturas, motor/modelo de OCR, modelo de IA, chave da API,
quantidade de threads, renomeação com código único, organização em
pastas (ano/projeto) e gravação de metadados EXIF.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Optional

import config as app_config
from interface.temas import aplicar_tema


class DialogoConfiguracoes(tk.Toplevel):
    def __init__(self, master: tk.Tk, ao_salvar: Optional[Callable[[], None]] = None):
        super().__init__(master)
        self.title("Configurações — CAMP Vision")
        self.geometry("560x680")
        self.resizable(False, False)
        self.ao_salvar = ao_salvar

        self.settings = app_config.settings
        aplicar_tema(self, self.settings.tema)

        self._construir_interface()
        self.transient(master)
        self.grab_set()

    def _construir_interface(self) -> None:
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg=self["bg"])
        barra_rolagem = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        area_rolavel = ttk.Frame(canvas)
        area_rolavel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=area_rolavel, anchor="nw")
        canvas.configure(yscrollcommand=barra_rolagem.set)
        canvas.pack(side="left", fill="both", expand=True)
        barra_rolagem.pack(side="right", fill="y")

        linha = 0

        def campo_texto(rotulo: str, valor_inicial: str, largura: int = 30) -> tk.StringVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=valor_inicial)
            ttk.Entry(area_rolavel, textvariable=var, width=largura).grid(row=linha, column=1, sticky="w", pady=4)
            linha += 1
            return var

        def campo_combo(rotulo: str, opcoes: list[str], valor_inicial: str) -> tk.StringVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=valor_inicial)
            combo = ttk.Combobox(area_rolavel, textvariable=var, values=opcoes, state="readonly", width=27)
            combo.grid(row=linha, column=1, sticky="w", pady=4)
            linha += 1
            return var

        def campo_numero(rotulo: str, valor_inicial: int, minimo: int, maximo: int) -> tk.IntVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.IntVar(value=valor_inicial)
            ttk.Spinbox(area_rolavel, from_=minimo, to=maximo, textvariable=var, width=8).grid(
                row=linha, column=1, sticky="w", pady=4)
            linha += 1
            return var

        def campo_decimal(rotulo: str, valor_inicial: float, minimo: float, maximo: float, incremento: float = 0.05) -> tk.DoubleVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.DoubleVar(value=valor_inicial)
            ttk.Spinbox(area_rolavel, from_=minimo, to=maximo, increment=incremento, textvariable=var, width=8).grid(
                row=linha, column=1, sticky="w", pady=4)
            linha += 1
            return var

        def campo_checkbox(rotulo: str, valor_inicial: bool) -> tk.BooleanVar:
            nonlocal linha
            var = tk.BooleanVar(value=valor_inicial)
            ttk.Checkbutton(area_rolavel, text=rotulo, variable=var).grid(
                row=linha, column=0, columnspan=2, sticky="w", pady=4)
            linha += 1
            return var

        def campo_arquivo(rotulo: str, valor_inicial: str, tipos: list[tuple[str, str]]) -> tk.StringVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=valor_inicial)
            sub = ttk.Frame(area_rolavel)
            sub.grid(row=linha, column=1, sticky="w", pady=4)
            ttk.Entry(sub, textvariable=var, width=22).pack(side="left")
            ttk.Button(sub, text="...", width=3, command=lambda: self._selecionar_arquivo(var, tipos)).pack(
                side="left", padx=(4, 0))
            linha += 1
            return var

        def campo_pasta(rotulo: str, valor_inicial: str) -> tk.StringVar:
            nonlocal linha
            ttk.Label(area_rolavel, text=rotulo).grid(row=linha, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=valor_inicial)
            sub = ttk.Frame(area_rolavel)
            sub.grid(row=linha, column=1, sticky="w", pady=4)
            ttk.Entry(sub, textvariable=var, width=22).pack(side="left")
            ttk.Button(sub, text="...", width=3, command=lambda: self._selecionar_pasta(var)).pack(
                side="left", padx=(4, 0))
            linha += 1
            return var

        def separador(titulo: str) -> None:
            nonlocal linha
            ttk.Label(area_rolavel, text=titulo, font=("", 11, "bold")).grid(
                row=linha, column=0, columnspan=2, sticky="w", pady=(14, 4))
            linha += 1

        # --- Geral ---
        separador("Geral")
        self.var_idioma = campo_combo("Idioma:", ["pt-BR", "en-US"], self.settings.idioma)
        self.var_pasta_padrao = campo_pasta("Pasta padrão:", self.settings.pasta_padrao)
        self.var_tema = campo_combo("Tema:", ["claro", "escuro"], self.settings.tema)
        self.var_threads = campo_numero("Quantidade de threads:", self.settings.quantidade_threads, 1, 64)

        # --- Miniaturas ---
        separador("Miniaturas")
        self.var_miniatura_tamanho = campo_numero(
            "Tamanho da miniatura (px):", self.settings.miniatura_tamanho_px, 200, 5000)
        self.var_miniatura_qualidade = campo_numero(
            "Qualidade JPEG (%):", self.settings.miniatura_qualidade, 10, 100)

        # --- OCR ---
        separador("OCR")
        self.var_ocr_motor = campo_combo("Motor de OCR:", ["tesseract", "paddleocr"], self.settings.ocr_motor)
        ttk.Label(
            area_rolavel, text="'tesseract' é recomendado em Macs mais antigos.",
            foreground="#8e8e93",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1

        # --- IA ---
        separador("Interpretação por IA")
        self.var_ia_modelo = campo_combo(
            "Modelo de IA:", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], self.settings.ia_modelo)
        self.var_ia_habilitada = campo_checkbox("Habilitar interpretação por IA", self.settings.ia_habilitada)
        self.var_api_key = campo_texto("Chave da API (OpenAI):", self.settings.ia_api_key)

        # --- Detecção/classificação por modelo treinado ---
        separador("Detecção e classificação (avançado)")
        self.var_deteccao_modo = campo_combo(
            "Detecção de carimbo:", ["heuristico", "modelo_treinado"], self.settings.deteccao_carimbo_modo)
        self.var_carimbo_regiao = campo_combo(
            "Região do carimbo:",
            ["automatico", "superior_esquerdo", "superior_direito", "inferior_esquerdo", "inferior_direito",
             "faixa_direita", "faixa_inferior", "faixa_esquerda", "faixa_superior"],
            self.settings.carimbo_regiao_busca,
        )
        ttk.Label(
            area_rolavel,
            text="Fixar a região (ex.: 'faixa_direita') reduz falsos positivos\nse o carimbo está sempre no mesmo lugar no projeto.",
            foreground="#8e8e93", justify="left",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1
        self.var_modelo_carimbo = campo_arquivo(
            "Modelo treinado (carimbo):", self.settings.caminho_modelo_carimbo, [("Pesos YOLO", "*.pt")])
        self.var_confianca_ml = campo_decimal(
            "Confiança mínima (modelo treinado):", self.settings.confianca_minima_ml, 0.0, 1.0)
        self.var_tamanho_imagem_ml = campo_numero(
            "Tamanho de imagem (0 = automático):", self.settings.tamanho_imagem_ml, 0, 1920)
        ttk.Label(
            area_rolavel,
            text="0 detecta automaticamente lendo o arquivo do modelo treinado\n"
                 "(recomendado — não precisa saber esse número). Só mude se\n"
                 "tiver certeza de que precisa forçar um valor específico.",
            foreground="#8e8e93", justify="left",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1
        self.var_classificacao_modo = campo_combo(
            "Classificação de tipo:", ["regras", "modelo_treinado"], self.settings.classificacao_modo)
        self.var_modelo_classificacao = campo_arquivo(
            "Modelo treinado (tipo):", self.settings.caminho_modelo_classificacao, [("Modelo", "*.pt")])

        # --- Renomeação ---
        separador("Renomeação de arquivos")
        self.var_renomeacao = campo_checkbox(
            "Renomear arquivos originais com código único", self.settings.renomeacao_habilitada)
        self.var_padrao_renomeacao = campo_texto("Padrão de nomenclatura:", self.settings.renomeacao_padrao, 30)
        ttk.Label(
            area_rolavel,
            text="Campos disponíveis: {codigo_projeto_auto} {sequencial_no_projeto}\n"
                 "{projeto} {nome_prancha} {codigo_projeto} {folha} {sequencial} {tipo}\n"
                 "{ano} {arquiteto} {endereco} {cliente} {fase} {prancha}\n"
                 "codigo_projeto_auto e sequencial_no_projeto são atribuídos\n"
                 "automaticamente por projeto (ex.: \"OCG-P0001\", \"N0012\"), com\n"
                 "prefixo tirado do arquiteto/cidade do carimbo. Campos vazios\n"
                 "são removidos do nome automaticamente.",
            foreground="#8e8e93", justify="left",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1
        self.var_digitos_sequencial = campo_numero(
            "Dígitos do sequencial:", self.settings.renomeacao_digitos_sequencial, 3, 10)

        # --- Organização em pastas (CONARQ) ---
        separador("Organização em pastas (CONARQ)")
        self.var_arquivamento = campo_checkbox(
            "Organizar em pastas por ano/projeto", self.settings.arquivamento_habilitado)
        self.var_padrao_pastas = campo_texto("Padrão de pastas:", self.settings.arquivamento_padrao_pastas, 30)
        ttk.Label(
            area_rolavel,
            text="Campos disponíveis: {ano} {codigo_projeto_auto} {projeto}\n"
                 "Por padrão, uma pasta por PROJETO detectado (não pelo nome\n"
                 "da pasta selecionada) — útil quando o lote tem vários\n"
                 "projetos diferentes digitalizados juntos.",
            foreground="#8e8e93", justify="left",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1
        self.var_raiz_arquivamento = campo_pasta(
            "Pasta raiz (vazio = pasta selecionada):", self.settings.arquivamento_pasta_raiz)
        self.var_arquivamento_copiar = campo_checkbox(
            "Copiar em vez de mover (mantém o arquivo original no lugar)",
            self.settings.arquivamento_copiar,
        )

        # --- Metadados EXIF ---
        separador("Metadados no arquivo")
        self.var_exif = campo_checkbox(
            "Gravar metadados no arquivo (EXIF/IPTC, via exiftool)", self.settings.gravar_metadados_exif)
        self.var_atribuicao_instituicao = campo_texto(
            "Atribuição no Copyright:", self.settings.atribuicao_instituicao, 40)
        ttk.Label(
            area_rolavel,
            text="Gravado no campo Copyright/Rights junto com o nome do\n"
                 "arquiteto, no formato \"arquiteto / atribuição\" (deixe em\n"
                 "branco para gravar só o arquiteto).",
            foreground="#8e8e93", justify="left",
        ).grid(row=linha, column=0, columnspan=2, sticky="w")
        linha += 1

        # --- Botões ---
        linha_botoes = ttk.Frame(area_rolavel)
        linha_botoes.grid(row=linha, column=0, columnspan=2, sticky="w", pady=(16, 0))
        ttk.Button(linha_botoes, text="Salvar", command=self._salvar).pack(side="left")
        ttk.Button(linha_botoes, text="Cancelar", command=self.destroy).pack(side="left", padx=(8, 0))

    def _selecionar_pasta(self, var: tk.StringVar) -> None:
        pasta = filedialog.askdirectory(title="Selecionar pasta", initialdir=var.get() or None)
        if pasta:
            var.set(pasta)

    def _selecionar_arquivo(self, var: tk.StringVar, tipos: list[tuple[str, str]]) -> None:
        caminho = filedialog.askopenfilename(title="Selecionar arquivo", filetypes=tipos)
        if caminho:
            var.set(caminho)

    def _salvar(self) -> None:
        self.settings.idioma = self.var_idioma.get()
        self.settings.pasta_padrao = self.var_pasta_padrao.get()
        self.settings.tema = self.var_tema.get()
        self.settings.quantidade_threads = self.var_threads.get()
        self.settings.miniatura_tamanho_px = self.var_miniatura_tamanho.get()
        self.settings.miniatura_qualidade = self.var_miniatura_qualidade.get()
        self.settings.ocr_motor = self.var_ocr_motor.get()
        self.settings.ia_modelo = self.var_ia_modelo.get()
        self.settings.ia_habilitada = self.var_ia_habilitada.get()
        self.settings.ia_api_key = self.var_api_key.get()
        self.settings.deteccao_carimbo_modo = self.var_deteccao_modo.get()
        self.settings.caminho_modelo_carimbo = self.var_modelo_carimbo.get()
        self.settings.confianca_minima_ml = self.var_confianca_ml.get()
        self.settings.tamanho_imagem_ml = self.var_tamanho_imagem_ml.get()
        self.settings.carimbo_regiao_busca = self.var_carimbo_regiao.get()
        self.settings.classificacao_modo = self.var_classificacao_modo.get()
        self.settings.caminho_modelo_classificacao = self.var_modelo_classificacao.get()
        self.settings.renomeacao_habilitada = self.var_renomeacao.get()
        self.settings.renomeacao_padrao = (
            self.var_padrao_renomeacao.get()
            or "{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}"
        )
        self.settings.renomeacao_digitos_sequencial = self.var_digitos_sequencial.get()
        self.settings.arquivamento_habilitado = self.var_arquivamento.get()
        self.settings.arquivamento_padrao_pastas = (
            self.var_padrao_pastas.get() or "{ano}/{codigo_projeto_auto} - {projeto}"
        )
        self.settings.arquivamento_pasta_raiz = self.var_raiz_arquivamento.get()
        self.settings.arquivamento_copiar = self.var_arquivamento_copiar.get()
        self.settings.gravar_metadados_exif = self.var_exif.get()
        self.settings.atribuicao_instituicao = self.var_atribuicao_instituicao.get()

        app_config.salvar_configuracoes(self.settings)

        if self.ao_salvar:
            self.ao_salvar()
        self.destroy()
