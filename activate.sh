#!/usr/bin/env bash
# Simple activator for local virtual environment (.venv)
# Usage: ./activate.sh

# Определяем путь к каталогу проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
PYTHON_BIN="$VENV_PATH/bin/python"
PIP_BIN="$VENV_PATH/bin/pip"

# Проверяем, существует ли окружение
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Run '$PYTHON_BIN -m venv .venv' in the project directory first."
    exit 1
fi

# Активируем окружение
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

# Проверяем корректность python
if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ Python binary not found in: $PYTHON_BIN"
    exit 1
fi

echo "✅ Activated virtual environment:"
echo "  $VENV_PATH"
echo "Python path: $PYTHON_BIN"
echo "Pip path:    $PIP_BIN"
echo "Python version: $($PYTHON_BIN --version 2>&1)"
echo
echo "You are now inside the environment. To exit, run: deactivate"
