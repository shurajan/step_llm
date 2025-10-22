#!/usr/bin/env bash
# Simple activator for local virtual environment (.venv)
# Usage: ./activate.sh

# Определяем путь к каталогу, где находится этот скрипт
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"

# Проверяем, существует ли окружение
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Run 'python -m venv .venv' in the project directory first."
    exit 1
fi

# Активируем окружение
source "$VENV_PATH/bin/activate"

echo "✅ Activated virtual environment:"
echo "  $VIRTUAL_ENV"
echo "Python version: $(python --version 2>&1)"
echo
echo "You are now inside the environment. To exit, run: deactivate"
