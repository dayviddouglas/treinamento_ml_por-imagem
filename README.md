# Treinamento de Machine Learning por Imagem

Este repositório contém um projeto de treinamento de modelos de Machine Learning utilizando imagens. Neste contexto o treinamento foi realizado com imagens de tênis das marcas: Nike, Mizuno e Adidas. A plataforma [Teachable Machine](https://teachablemachine.withgoogle.com/) foi utilizada para o treinamento do presente modelo, disponizando os arquivos: `model.tflite` e `labels.txt`.

## Estrutura do Projeto

- `main.py`: Script principal para treinamento e avaliação do modelo.
- `model.tflite`: Modelo treinado em formato TensorFlow Lite.
- `labels.txt`: Arquivo contendo os rótulos das classes.

## Requisitos

- Python 3.x
- TensorFlow
- NumPy
- IDE de sua preferência(foi utilizado o Visual Studio Code para realização do projeto.)
- Câmera no dispositivo ao qual vai executar o script.

## Instalação

Clone o repositório e instale os pacotes necessários:

```bash
git clone https://github.com/dayviddouglas/treinamento_ml_por-imagem.git
cd treinamento_ml_por-imagem
```

## Criar um Ambiente Virtual

Crie um ambiente virtual:

1. Crie um ambiente virtual:

```bash
 python -m venv venv
```
2. Ative o ambiente virtual:

- Windows:

```bash
venv\Scripts\activate
```
- macOS e Linux:

```bash
source venv/bin/activate
```
3. Instale as dependências:

```bash
pip install -r requirements.txt
```
## Uso:

Para treinar o modelo, execute o script main.py:

```bash
python main.py
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e enviar pull requests.