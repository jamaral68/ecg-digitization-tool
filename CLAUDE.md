# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack e comandos

- **Python 3.12** gerenciado via **Poetry**. Sempre use `poetry run <cmd>` — não há `python` global.
  - Fallback de venv: `source ~/.cache/pypoetry/virtualenvs/ecg-digitization-tool-IzEWH0ZB-py3.12/bin/activate`
- Lint/format: **ruff** via pre-commit (`pre-commit run --all-files`). `nbstripout` remove outputs de notebooks no commit.
- **Não existe suíte de testes** neste repo. `test.py` na raiz é um script solto, não pytest.

### Make targets frequentes
- `make app` — Streamlit local em `:8501`
- `make demo` — Streamlit + túnel ngrok (requer `NGROK_AUTHTOKEN` em `.env`)
- `make build` / `make run` / `make push` / `make register` — pipeline Docker/SageMaker
- `make install-poetry` — bootstrap inicial

## Arquitetura — duas pipelines distintas

O projeto tem **dois caminhos de processamento independentes** que não se chamam entre si. Entender a distinção é crítico:

### 1. `src/ecg_scanner/` — pré-processamento de documento
Classe `ECGScanner` (com dataclass `ECGScannerConfig`). Objetivo: dada uma foto "torta" de um ECG impresso, detectar o retângulo do papel, aplicar transformação de perspectiva e binarizar. Pipeline clássico de visão computacional: CLAHE → bilateral filter → Canny → LSD (pylsd) → seleção de quadrilátero → `four_point_transform` → adaptive threshold. Estágios intermediários são salvos em `self.stages` quando `debug_mode=True` e visualizados via `src/utils.py::plot_stages`.

### 2. `src/ecg_digitizer/` — extração do sinal (YOLO)
Função `ecg_to_csv(config: DigitizerConfig, model, label_model)`. Objetivo: dado um ECG já endireitado, detectar cada lead via YOLO, remover textos sobrepostos por inpainting, e extrair a curva via `np.argmin` coluna-a-coluna no crop em grayscale. Converte pixels→(s, mV) com `convert_to_secmv` e reamostra com `CubicSpline`.
- Dois modelos YOLO: `models/best.pt` (leads + calibração "pulse") e `models/labels.pt` (textos).
- Calibração mV/pixel vem da altura do bbox da classe `pulse`; fallback = 10.
- Treino dos modelos **não está neste repo** — `best.pt`/`labels.pt` são artefatos herdados.

### 3. `app/` — UI Streamlit
- `app/app.py` — entrypoint; carrega YOLO com `@st.cache_resource` para não reimportar 12 MB a cada rerun.
- `app/serve_ngrok.py` — launcher que sobe Streamlit como subprocess + abre túnel ngrok para demos.
- Inputs da UI: `pulse_width_mm`, `mm_per_sec`, `sample_frequency`, `lead_time`. **Atenção:** `pulse_per_sec = pulse_width_mm / mm_per_sec` produz valores aparentemente inconsistentes com `lead_time` — há suspeita de bug de calibração ainda não resolvida. Não "consertar" silenciosamente.

### Imports — convenção sem `__init__.py`
Os subpacotes em `src/` (`ecg_scanner`, `ecg_digitizer`, `commons`) **não têm `__init__.py`**. O padrão do projeto é inserir `<repo>/src` no `sys.path` em tempo de execução:
- Notebooks: `sys.path.append(str(Path.cwd().parent / "src"))`
- `app/app.py`: `sys.path.insert(0, str(PROJECT_ROOT / "src"))`

Ao criar novo subpacote em `src/`, siga a mesma convenção (não adicione `__init__.py` só para organizar imports — mantém compatibilidade com o padrão atual).

### Convenção de naming de config
Cada pipeline expõe uma dataclass `*Config`: `ECGScannerConfig`, `DigitizerConfig`. Preserve esse padrão ao adicionar novas pipelines.

## Duplicação intencional: `pyproject.toml` raiz vs `deploy/pyproject.sagemaker.toml`

A raiz instala **PyTorch ROCm** (dev local em GPU AMD). O SageMaker roda em NVIDIA, então `deploy/pyproject.sagemaker.toml` carrega builds CUDA. Ao adicionar/atualizar deps, **replique nos dois arquivos** (exceto `torch`/`torchvision`, que divergem de propósito). Ver `deploy/README.md`.

## Artefatos fora do git
- `models/` — pesos YOLO (~12 MB cada), gitignored. Quem clona precisa obter por fora.
- `datasets/` — imagens de entrada, gitignored.
- `.env` — credenciais (S3, ngrok).

## Código legado
`deprecated/` contém implementações antigas e **é ignorado por todos os hooks do pre-commit**. Não usar como referência e não reativar sem limpar.
