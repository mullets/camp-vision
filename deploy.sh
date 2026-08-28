#!/usr/bin/env bash
# deploy.sh
# ==========
# Checa se há atualização no repositório, e se houver:
#   1. Espera não ter processamento em andamento (trava em
#      ~/.campvision/processando.lock)
#   2. git pull
#   3. Reinstala dependências SE requirements*.txt mudou
#   4. Reinicia o serviço systemd
#
# Pensado para rodar via systemd timer a cada N minutos (ver
# campvision-deploy.timer) — nunca interrompe um lote em processamento.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICO="campvision-automatizado"
TRAVA="$HOME/.campvision/processando.lock"
LOG_TAG="[deploy.sh]"

cd "$REPO_DIR"

echo "$LOG_TAG Verificando atualizações em $REPO_DIR..."
git fetch --quiet origin main

LOCAL=$(git rev-parse HEAD)
REMOTO=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTO" ]; then
    echo "$LOG_TAG Já está na versão mais recente ($LOCAL). Nada a fazer."
    exit 0
fi

echo "$LOG_TAG Nova versão disponível: $LOCAL -> $REMOTO"

# Nunca atualiza no meio de um lote — espera a trava sumir, com um
# limite de tentativas (não trava pra sempre se algo travou de verdade
# no processamento; nesse caso, só tenta de novo na próxima rodada do
# timer, sem forçar nada).
TENTATIVAS=0
MAX_TENTATIVAS=30  # 30 x 10s = 5 minutos de espera no máximo
while [ -f "$TRAVA" ] && [ "$TENTATIVAS" -lt "$MAX_TENTATIVAS" ]; do
    echo "$LOG_TAG Processamento em andamento, aguardando... (tentativa $((TENTATIVAS+1))/$MAX_TENTATIVAS)"
    sleep 10
    TENTATIVAS=$((TENTATIVAS+1))
done

if [ -f "$TRAVA" ]; then
    echo "$LOG_TAG Ainda em processamento após 5 minutos de espera — adiando o deploy para a próxima verificação."
    exit 0
fi

echo "$LOG_TAG Atualizando código (git pull)..."
git pull --quiet origin main

REQUIREMENTS_MUDOU=false
if git diff --name-only "$LOCAL" "$REMOTO" | grep -qE "^requirements.*\.txt$"; then
    REQUIREMENTS_MUDOU=true
fi

if [ "$REQUIREMENTS_MUDOU" = true ]; then
    echo "$LOG_TAG requirements.txt mudou — reinstalando dependências..."
    if ! "$REPO_DIR/.venv/bin/pip" install --quiet -r requirements.txt; then
        echo "$LOG_TAG ERRO ao instalar dependências novas. NÃO reiniciando o serviço"
        echo "$LOG_TAG (o serviço continua rodando com o código ANTERIOR até isso ser corrigido manualmente)."
        exit 1
    fi
fi

echo "$LOG_TAG Reiniciando o serviço $SERVICO..."
sudo systemctl restart "$SERVICO"
echo "$LOG_TAG Deploy concluído: agora em $REMOTO"
