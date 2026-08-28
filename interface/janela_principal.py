"""
interface/janela_principal.py
==============================
Janela principal do CAMP Vision (Tkinter): código do projeto, seleção
de pasta, início/cancelamento do processamento, barra de progresso,
log em tempo real, contagem de arquivos processados, tempo estimado e
acesso às configurações.

Escolhemos Tkinter em vez de um framework como PySide6/Qt de propósito:
Tkinter é parte da biblioteca padrão do Python (nenhuma dependência
binária pesada para instalar) e funciona em versões de macOS e
hardware bem mais antigos — importante para escritórios que ainda
usam Macs mais velhos (ex.: Mac Pro 5,1/6,1), onde frameworks Qt
recentes podem nem sequer instalar.

O processamento roda em uma thread separada (não trava a interface),
e a comunicação de progresso/log ocorre por meio de filas
(`queue.Queue`), consumidas periodicamente pelo loop principal do
Tkinter via `root.after(...)` — Tkinter não é thread-safe para
atualização direta de widgets a partir de outras threads.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import config as app_config
from database.models import criar_engine
from interface.dialogo_configuracoes import DialogoConfiguracoes
from interface.temas import aplicar_tema
from scanner.lote import ConfiguracaoLote, ProcessadorLote, ProgressoLote
from utils.logger import configurar_logging
from utils.renomeador import sugerir_codigo_projeto

INTERVALO_ATUALIZACAO_MS = 200


class JanelaPrincipal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{app_config.APP_NAME} — build {app_config.VERSAO_BUILD} — Catalogação automática de acervos")
        self.geometry("900x650")
        self.minsize(760, 560)

        self.pasta_selecionada: Path | None = None
        self.processador_atual: ProcessadorLote | None = None
        self.thread_processamento: threading.Thread | None = None

        self.fila_log: "queue.Queue[str]" = queue.Queue()
        self.fila_progresso: "queue.Queue[ProgressoLote]" = queue.Queue()
        self.fila_conclusao: "queue.Queue[tuple]" = queue.Queue()

        self.logger = configurar_logging(app_config.LOG_DIR, gui_queue=self.fila_log)

        self.paleta = aplicar_tema(self, app_config.settings.tema)
        self._construir_interface()

        self.after(INTERVALO_ATUALIZACAO_MS, self._consumir_filas)

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------
    def _construir_interface(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        # --- Pasta do acervo ---
        grupo_pasta = ttk.LabelFrame(container, text="Pasta do acervo", padding=8)
        grupo_pasta.pack(fill="x", pady=(0, 8))
        linha_pasta = ttk.Frame(grupo_pasta)
        linha_pasta.pack(fill="x")
        self.var_pasta = tk.StringVar(value="Nenhuma pasta selecionada...")
        ttk.Entry(linha_pasta, textvariable=self.var_pasta, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(linha_pasta, text="Selecionar Pasta", command=self._selecionar_pasta).pack(side="left")
        self.var_aviso_pasta = tk.StringVar(value="")
        self.label_aviso_pasta = ttk.Label(grupo_pasta, textvariable=self.var_aviso_pasta, foreground="#ff3b30")
        self.label_aviso_pasta.pack(fill="x", pady=(4, 0))

        # --- Código do projeto ---
        grupo_codigo = ttk.LabelFrame(container, text="Código do projeto", padding=8)
        grupo_codigo.pack(fill="x", pady=(0, 8))
        self.var_codigo_projeto = tk.StringVar()
        ttk.Entry(grupo_codigo, textvariable=self.var_codigo_projeto, width=20).pack(side="left", padx=(0, 8))
        ttk.Label(grupo_codigo, text='Gera nomes como: SB-P001.1-00001.tif', foreground="#8e8e93").pack(side="left")

        # --- Ações principais ---
        grupo_acoes = ttk.Frame(container)
        grupo_acoes.pack(fill="x", pady=(0, 8))
        self.botao_processar = ttk.Button(grupo_acoes, text="▶  Processar", command=self._iniciar_processamento)
        self.botao_processar.pack(side="left")
        self.botao_cancelar = ttk.Button(
            grupo_acoes, text="■  Cancelar", command=self._cancelar_processamento, state="disabled")
        self.botao_cancelar.pack(side="left", padx=(8, 0))
        ttk.Button(grupo_acoes, text="⚙  Configurações", command=self._abrir_configuracoes).pack(side="right")

        # --- Progresso ---
        grupo_progresso = ttk.LabelFrame(container, text="Progresso", padding=8)
        grupo_progresso.pack(fill="x", pady=(0, 8))
        self.barra_progresso = ttk.Progressbar(grupo_progresso, mode="determinate", maximum=100)
        self.barra_progresso.pack(fill="x", pady=(0, 6))

        linha_info = ttk.Frame(grupo_progresso)
        linha_info.pack(fill="x")
        self.var_contagem = tk.StringVar(value="0 / 0 arquivos processados")
        self.var_tempo_restante = tk.StringVar(value="Tempo estimado restante: —")
        ttk.Label(linha_info, textvariable=self.var_contagem).pack(side="left")
        ttk.Label(linha_info, textvariable=self.var_tempo_restante).pack(side="right")

        self.var_arquivo_atual = tk.StringVar(value="Arquivo atual: —")
        ttk.Label(grupo_progresso, textvariable=self.var_arquivo_atual).pack(fill="x", pady=(4, 0))

        # --- Log em tempo real ---
        grupo_log = ttk.LabelFrame(container, text="Log em tempo real", padding=8)
        grupo_log.pack(fill="both", expand=True)
        self.texto_log = ScrolledText(
            grupo_log, wrap="none", state="disabled", height=14,
            bg=self.paleta["fundo_widget"], fg=self.paleta["texto"],
            insertbackground=self.paleta["texto"], relief="flat",
        )
        self.texto_log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def _selecionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(
            title="Selecionar pasta do acervo", initialdir=app_config.settings.pasta_padrao)
        if pasta:
            self.pasta_selecionada = Path(pasta)
            self.var_pasta.set(pasta)
            if not self.var_codigo_projeto.get().strip():
                self.var_codigo_projeto.set(sugerir_codigo_projeto(self.pasta_selecionada.name))
            if self._parece_pasta_de_saida(self.pasta_selecionada):
                self.var_aviso_pasta.set(
                    "⚠ Essa pasta parece ser uma SAÍDA de processamento anterior "
                    "(\"_catalogado\" no caminho), não a pasta original."
                )
            else:
                self.var_aviso_pasta.set("")

    @staticmethod
    def _parece_pasta_de_saida(pasta: Path) -> bool:
        """Detecta se a pasta selecionada parece ser uma pasta de
        SAÍDA gerada pelo próprio CAMP Vision (a estrutura
        ano/projeto criada por `utils.arquivamento`, ou a pasta de
        relatórios `catalogacao_saida`) em vez da pasta original com
        os arquivos crus — evita reprocessar/aninhar por engano."""
        partes = pasta.parts
        return any(p.endswith("_catalogado") for p in partes) or "catalogacao_saida" in partes

    def _iniciar_processamento(self) -> None:
        if self.pasta_selecionada is None:
            messagebox.showwarning("Nenhuma pasta selecionada", "Selecione uma pasta antes de processar.")
            return

        if self._parece_pasta_de_saida(self.pasta_selecionada):
            confirmar = messagebox.askyesno(
                "Isso parece uma pasta de SAÍDA",
                "A pasta selecionada tem \"_catalogado\" ou \"catalogacao_saida\" no "
                "caminho — parece ser uma pasta gerada por um processamento anterior, "
                "não a pasta original com os TIFFs crus.\n\n"
                "Processar aqui de novo pode reprocessar arquivos já feitos e aninhar "
                "pastas.\n\n"
                "Tem certeza que quer continuar mesmo assim?",
                icon="warning",
            )
            if not confirmar:
                return

        codigo_projeto = self.var_codigo_projeto.get().strip()
        if app_config.settings.renomeacao_habilitada and not codigo_projeto:
            messagebox.showwarning(
                "Código do projeto necessário",
                "Informe um código do projeto (ex.: SB) antes de processar, "
                "ou desabilite a renomeação nas Configurações.",
            )
            return

        # Tudo o que o app produz (CSV/XLSX/JSON, recortes de carimbo,
        # miniaturas E as pranchas arquivadas) fica junto, na raiz da
        # pasta catalogada — em vez de espalhar relatórios dentro da
        # pasta original do acervo, que deve permanecer intocada.
        pasta_saida = self.pasta_selecionada.parent / f"{self.pasta_selecionada.name}_catalogado"
        criar_engine(str(app_config.DB_PATH))  # garante que o schema exista

        config_lote = ConfiguracaoLote(
            pasta_entrada=self.pasta_selecionada,
            pasta_saida=pasta_saida,
            formatos_aceitos=app_config.settings.formatos_aceitos,
            quantidade_threads=app_config.settings.quantidade_threads,
            idiomas_ocr=app_config.settings.ocr_idiomas,
            tamanho_miniatura=app_config.settings.miniatura_tamanho_px,
            qualidade_miniatura=app_config.settings.miniatura_qualidade,
            salvar_miniaturas=app_config.settings.salvar_miniaturas,
            salvar_carimbos=app_config.settings.salvar_carimbos,
            ia_api_key=app_config.settings.ia_api_key,
            ia_modelo=app_config.settings.ia_modelo,
            ia_habilitada=app_config.settings.ia_habilitada,
            caminho_db=str(app_config.DB_PATH),
            ocr_motor=app_config.settings.ocr_motor,
            deteccao_carimbo_modo=app_config.settings.deteccao_carimbo_modo,
            caminho_modelo_carimbo=app_config.settings.caminho_modelo_carimbo,
            carimbo_regiao_busca=app_config.settings.carimbo_regiao_busca,
            classificacao_modo=app_config.settings.classificacao_modo,
            caminho_modelo_classificacao=app_config.settings.caminho_modelo_classificacao,
            confianca_minima_ml=app_config.settings.confianca_minima_ml,
            tamanho_imagem_ml=app_config.settings.tamanho_imagem_ml,
            codigo_projeto=codigo_projeto,
            renomeacao_habilitada=app_config.settings.renomeacao_habilitada,
            renomeacao_padrao=app_config.settings.renomeacao_padrao,
            renomeacao_digitos_sequencial=app_config.settings.renomeacao_digitos_sequencial,
            gravar_metadados_exif=app_config.settings.gravar_metadados_exif,
            arquivamento_habilitado=app_config.settings.arquivamento_habilitado,
            arquivamento_padrao_pastas=app_config.settings.arquivamento_padrao_pastas,
            arquivamento_pasta_raiz=(
                Path(app_config.settings.arquivamento_pasta_raiz)
                if app_config.settings.arquivamento_pasta_raiz else None
            ),
            arquivamento_copiar=app_config.settings.arquivamento_copiar,
            deteccao_multiorientacao=app_config.settings.deteccao_multiorientacao,
            nome_projeto=self.pasta_selecionada.name,
            atribuicao_instituicao=app_config.settings.atribuicao_instituicao,
        )

        self.processador_atual = ProcessadorLote(config_lote, callback_progresso=self.fila_progresso.put_nowait)

        self.botao_processar.configure(state="disabled")
        self.botao_cancelar.configure(state="normal")
        self.texto_log.configure(state="normal")
        self.texto_log.delete("1.0", "end")
        self.texto_log.configure(state="disabled")

        self.thread_processamento = threading.Thread(target=self._executar_em_thread, daemon=True)
        self.thread_processamento.start()

    def _executar_em_thread(self) -> None:
        """Executa o processamento em uma thread separada da interface
        e publica o resultado final na fila de conclusão."""
        try:
            resultados = self.processador_atual.executar()
            sucesso = sum(1 for r in resultados if r.sucesso)
            falhas = sum(1 for r in resultados if not r.sucesso)
            self.fila_conclusao.put_nowait(("ok", sucesso, falhas))
        except Exception as exc:  # noqa: BLE001
            self.fila_conclusao.put_nowait(("erro", str(exc), None))

    def _cancelar_processamento(self) -> None:
        if self.processador_atual is not None:
            self.processador_atual.cancelar()
        self.botao_cancelar.configure(state="disabled")

    def _abrir_configuracoes(self) -> None:
        DialogoConfiguracoes(self, ao_salvar=self._reaplicar_tema)

    def _reaplicar_tema(self) -> None:
        self.paleta = aplicar_tema(self, app_config.settings.tema)
        self.texto_log.configure(bg=self.paleta["fundo_widget"], fg=self.paleta["texto"])

    # ------------------------------------------------------------------
    # Consumo periódico das filas (progresso, log, conclusão)
    # ------------------------------------------------------------------
    def _consumir_filas(self) -> None:
        self._consumir_fila_progresso()
        self._consumir_fila_log()
        self._consumir_fila_conclusao()
        self.after(INTERVALO_ATUALIZACAO_MS, self._consumir_filas)

    def _consumir_fila_progresso(self) -> None:
        ultimo: ProgressoLote | None = None
        while True:
            try:
                ultimo = self.fila_progresso.get_nowait()
            except queue.Empty:
                break
        if ultimo is not None:
            percentual = (ultimo.concluidos / ultimo.total * 100) if ultimo.total else 0
            self.barra_progresso["value"] = percentual
            self.var_contagem.set(
                f"{ultimo.concluidos} / {ultimo.total} arquivos processados "
                f"({ultimo.sucesso} ok, {ultimo.falhas} com erro)"
            )
            self.var_tempo_restante.set(f"Tempo estimado restante: {ultimo.tempo_restante_formatado}")
            self.var_arquivo_atual.set(f"Arquivo atual: {ultimo.arquivo_atual}")

    def _consumir_fila_log(self) -> None:
        linhas_novas = []
        while True:
            try:
                linhas_novas.append(self.fila_log.get_nowait())
            except queue.Empty:
                break
        if linhas_novas:
            self.texto_log.configure(state="normal")
            self.texto_log.insert("end", "\n".join(linhas_novas) + "\n")
            self.texto_log.see("end")
            self.texto_log.configure(state="disabled")

    def _consumir_fila_conclusao(self) -> None:
        try:
            tipo, a, b = self.fila_conclusao.get_nowait()
        except queue.Empty:
            return

        self.botao_processar.configure(state="normal")
        self.botao_cancelar.configure(state="disabled")

        if tipo == "ok":
            sucesso, falhas = a, b
            messagebox.showinfo(
                "Processamento concluído",
                f"Concluído: {sucesso} arquivos processados com sucesso, {falhas} com erro.\n"
                f"Resultados salvos em: {self.pasta_selecionada.parent / (self.pasta_selecionada.name + '_catalogado')}",
            )
        else:
            messagebox.showerror("Erro no processamento", str(a))
