#!/bin/bash

source .venv/bin/activate  

run_notebook() {
    local notebook_file="$1"
    echo "$notebook_file запущен"
    jupyter nbconvert --to notebook --execute "$notebook_file"  --stdout > /dev/null 2>&1
    echo "$notebook_file завершён"
}

run_notebook "trend.ipynb"
run_notebook "separation.ipynb"
run_notebook "Andrews.ipynb"

deactivate
