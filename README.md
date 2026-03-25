# Ecg-digitization-tool

## Resumo 
Este projeto tem como objetivo converter imagens de eletrocardiogramas (ECG) em dados estruturados no formato CSV, permitindo análise digital de sinais cardíacos a partir de imagens.

## Instalação

1. **Clone o repositório**:

```bash

git clone -b matheus https://github.com/jamaral68/ecg-digitization-tool.git
cd ecg-digitization-tool

```

2. **Instale as dependências**:

```bash

pip install -r requirements.txt

```

3. **Execute o script principal**:

```bash

python main.py 

```

## Configuração das Variáveis

| Variável | Descrição |
|----------|-----------|
| `image` | Caminho da imagem do ECG |
| `template_name` | Imagem do pulso de referência |
| `csv_name` | Arquivo CSV de saída |
| `strategy` | Estratégia de pré-processamento (`none`, `filter`, `color`) |
| `thres_value` | Limiar de binarização |
| `dilation` | Número de iterações de dilatação |
| `perc_space_leads` | Espaçamento relativo entre leads |
| `layout` | Layout do ECG (linhas x colunas) |
| `rhythm` | Linha do traçado de ritmo |
| `pulse` | Linhas que contêm pulsos |
| `pulse_width_mm` | Largura do pulso em mm |
| `pulse_height_mm` | Altura do pulso em mm |
| `mmpsec` | Escala de tempo (mm por segundo) |
| `mmpmv` | Escala de amplitude (mm por mV) |
| `pulse_per_sec` | Largura do pulso em segundos |
| `pulse_per_mv` | Altura do pulso em mV |
| `sample_frequency` | Pontos por segundo (Hz) |
| `time_lead` | Duração do segmento de ECG em segundos |
| `num_sampling_points` | Número de pontos por lead |
| `location` | Local do pulso de referência (`left` ou `right`) |
| `lower` | Limite inferior de cor (HSV) |
| `upper` | Limite superior de cor (HSV) |
| `kSize2d` | Tamanho do kernel 2D para filtros |
| `kSize1d` | Tamanho do kernel 1D para filtros |