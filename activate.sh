#!/usr/bin/env bash

# перейти в директорию скрипта (корень проекта)
cd "$(dirname "$0")"

# проверяем, что .venv существует
if [ ! -d ".venv" ]; then
    echo "Error: .venv not found in project root."
    exit 1
fi

# активируем виртуальное окружение
source .venv/bin/activate

echo "Activated .venv (Python: $(python --version))"
