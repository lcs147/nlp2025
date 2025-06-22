Siga os passos abaixo para configurar o ambiente e executar o pipeline experimental.

## 1. Pré-requisitos

- **Python 3.12** (recomendado)
- **Git**
- **GPU** com pelo menos 15GB de memória (ex: Google Colab T4) para fine-tuning. Para avaliação, CPU é suficiente (mais lento).

## 2. Clonando o Repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd fine-tuning_tradeoff
```

## 3. Instalando as Dependências

O projeto utiliza ambiente virtual para garantir versões exatas dos pacotes.

**Usando pip:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Ou usando pipenv:**
```bash
pip install pipenv
pipenv install
pipenv shell
```

> **Obs:** Se for utilizar GPU, certifique-se de que os drivers CUDA estejam instalados e compatíveis com as versões do PyTorch e bitsandbytes.

## 4. Download dos Dados

- Baixe o **Spider dataset** e coloque os arquivos em `datasets/spider_data/`.
- Os dados do **MMLU** são baixados automaticamente via Hugging Face Datasets.

## 5. Execução dos Scripts

Execute os scripts na ordem abaixo para reproduzir o pipeline:

1. **Pré-processamento dos dados:**
    ```bash
    python scripts/0_preparando_dados.py
    ```
    (Gera arquivos processados em `datasets/`)

2. **Fine-tuning supervisionado com LoRA:**
    ```bash
    python scripts/1_fine-tuning.py
    ```
    (Checkpoints salvos em `checkpoints/loras/`)

3. **Cálculo do baseline na tarefa-alvo:**
    ```bash
    python scripts/2_calculando_baseline.py
    ```

4. **Avaliação customizada com DeepEval:**
    ```bash
    python scripts/3_avaliacao_customizada.py
    ```

5. **Avaliação de generalização no MMLU:**
    ```bash
    python scripts/4_analise_mmlu.py
    ```

## 6. Resultados
Os resultados intermediários serão salvos na pasta `checkpoints/`.

Os resultados finais (acurácia, desvio padrão) são reportados na saida (stdout) dos scripts.

## 7. Observações

- Todos os seeds aleatórios foram fixados em 42 para garantir reprodutibilidade.
- Se utilizar Google Colab, adapte os caminhos dos arquivos conforme necessário.