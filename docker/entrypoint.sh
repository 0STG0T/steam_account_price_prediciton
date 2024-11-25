#!/bin/bash

case "$1" in
  "train")
    jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
    ;;
  "predict")
    python -m main.onnx_usage_example
    ;;
  *)
    echo "Usage: $0 {train|predict}"
    exit 1
    ;;
esac
