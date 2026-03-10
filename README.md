# Heart Disease Prediction

Pipeline de machine learning para predição de doença cardíaca usando regressão logística com rastreamento MLflow.

## Estrutura do Projeto

```
heart_disease/
├── data/raw/                 # Dados brutos
├── config/                   # Schemas de validação
├── heart_disease/           
│   ├── constants.py         # Constantes e configurações do projeto
│   ├── api/                 # API FastAPI
│   └── pipelines/           
│       ├── train.py         # TrainingPipeline class
│       ├── predict.py       # PredictionPipeline class
│       └── components/      
│           ├── dataset.py   # Carregamento e validação de dados
│           └── features.py  # Transformação de features
```

## Instalação

```bash
pip install -e .
```

## Pipeline de Treinamento

A pipeline de treinamento é implementada como a classe **`TrainingPipeline`** com métodos modulares e testáveis.

### Uso Básico

```python
from heart_disease.pipelines.train import TrainingPipeline

# Instanciar e executar
pipeline = TrainingPipeline()
results = pipeline.run()

# Acessar resultados
print(f"Run ID: {results['run_id']}")
print(f"ROC-AUC: {results['metrics']['test_roc_auc']:.4f}")
print(f"Modelo promovido: {results['promoted']}")

# Acessar atributos da pipeline após treinamento
print(f"Modelo treinado: {pipeline.model_}")
print(f"Métricas: {pipeline.metrics_}")
print(f"MLflow Run ID: {pipeline.run_id_}")
```

### Parâmetros de Configuração

```python
# Com parâmetros customizados
pipeline = TrainingPipeline(
    n_iter=100,           # Iterações de busca aleatória
    cv_folds=10,          # Folds de validação cruzada
    force_replace=True,   # Forçar substituição do modelo ativo
    data_path='custom/path/to/data.csv'  # Caminho customizado de dados
)
results = pipeline.run()
```

### Arquitetura da Classe

**`TrainingPipeline`** organiza o workflow em métodos privados:

**Preparação de Dados:**
- `_load_and_validate_data()` - Carrega e valida dados
- `_prepare_features()` - Transforma e separa features/target
- `_split_train_test()` - Split estratificado

**Modelagem:**
- `_create_ml_pipeline()` - Cria pipeline sklearn
- `_tune_hyperparameters()` - Busca aleatória de hiperparâmetros
- `_evaluate_model()` - Calcula métricas

**MLflow:**
- `_log_to_mlflow()` - Registra modelo e métricas
- `_get_active_model_metric()` - Obtém métrica do modelo ativo
- `_should_promote_model()` - Decide sobre promoção
- `_promote_model()` - Promove modelo para "active"

**Orquestração:**
- `run()` - Método público que coordena todo o fluxo

### Atributos da Instância

Após executar `run()`, os seguintes atributos ficam disponíveis:

- `model_`: Pipeline sklearn treinado
- `metrics_`: Dicionário com métricas de avaliação
- `run_id_`: MLflow run ID

### Função de Compatibilidade

Para compatibilidade com código existente, a função `train_pipeline()` continua disponível:

```python
from heart_disease.pipelines.train import train_pipeline

# Uso funcional (cria e executa TrainingPipeline internamente)
results = train_pipeline(n_iter=100, cv_folds=10, force_replace=False)
```

### Testando Componentes Individuais

```python
from heart_disease.pipelines.train import TrainingPipeline

# Criar instância
pipeline = TrainingPipeline(n_iter=50, cv_folds=5)

# Testar métodos individuais (útil para testes unitários)
df = pipeline._load_and_validate_data()
X, y = pipeline._prepare_features(df)
ml_pipeline = pipeline._create_ml_pipeline(X)

# Ou executar o workflow completo
results = pipeline.run()
```

## Configuração

Todas as configurações estão centralizadas em [`heart_disease/constants.py`](heart_disease/constants.py):

- **Caminhos**: diretórios de dados e arquivos
- **Colunas**: features numéricas, categóricas, target
- **Modelagem**: random_state, test_size, cv_splits
- **MLflow**: model_name, artifact_path, alias
- **Hiperparâmetros**: grid de busca para regressão logística
- **Defaults**: n_iter, cv_folds, scoring_metric, n_jobs

## MLflow Tracking

A pipeline registra automaticamente:

- **Parâmetros**: configurações de treinamento e melhores hiperparâmetros
- **Métricas**: accuracy, precision, recall, F1, ROC-AUC (CV e teste)
- **Modelo**: pipeline sklearn completo com assinatura de entrada/saída
- **Registro**: modelos versionados com alias "active" para produção

### Visualizar Experimentos

```bash
mlflow ui
```

Acesse http://localhost:5000 para visualizar experimentos, comparar modelos e gerenciar o registro.

## Gestão de Modelos

O modelo treinado é automaticamente promovido para "active" se:

1. **Melhor performance**: ROC-AUC no teste > modelo ativo atual, OU
2. **Forçar substituição**: `force_replace=True`

O modelo com alias "active" é usado pela API de produção para inferência.

## Pipeline de Predição

A pipeline de predição carrega o modelo ativo do MLflow e faz predições para novos pacientes. Reutiliza os componentes de validação e transformação de dados.

### Uso Básico

**Predições a partir de DataFrame:**

```python
from heart_disease.pipelines.predict import PredictionPipeline
import pandas as pd

# Criar dados de exemplo
patients = pd.DataFrame({
    'id': [1, 2],
    'Age': [55, 62],
    'Sex': [1, 0],  # 1=male, 0=female
    'Chest pain type': [2, 3],
    'BP': [130, 140],
    'Cholesterol': [240, 260],
    'FBS over 120': [1, 1],
    'EKG results': [0, 1],
    'Max HR': [150, 135],
    'Exercise angina': [0, 1],
    'ST depression': [1.5, 2.0],
    'Slope of ST': [2, 1],
    'Number of vessels fluro': [1, 2],
    'Thallium': [6, 7]
})

# Instanciar e carregar modelo
pipeline = PredictionPipeline()
pipeline.load_model()

# Fazer predições
results = pipeline.predict(patients)
print(results[['id', 'Age', 'prediction']])
```

**Predições com probabilidades:**

```python
# Incluir probabilidades das classes
results = pipeline.predict(patients, return_proba=True)
print(results[['id', 'prediction', 'probability_Presence', 'probability_Absence']])
```

**Predições a partir de arquivo CSV:**

```python
# Carregar e predizer de uma vez
results = pipeline.predict_from_file(
    'data/new_patients.csv',
    return_proba=True
)
```

### Função de Conveniência

Para predições rápidas sem instanciar a classe:

```python
from heart_disease.pipelines.predict import predict_patients

# De DataFrame
results = predict_patients(patients_df, return_proba=True)

# De arquivo CSV
results = predict_patients('data/new_patients.csv', return_proba=True)

# Sem incluir dados de entrada
results = predict_patients(patients_df, include_input=False)

# Usar modelo específico
results = predict_patients(
    patients_df,
    model_name='heart_disease_model',
    model_alias='champion'
)
```

### Parâmetros da Pipeline

**`PredictionPipeline`:**
- `model_name`: Nome do modelo no MLflow (default: do constants.py)
- `model_alias`: Alias da versão (default: "active")

**Método `predict()`:**
- `data`: DataFrame com dados dos pacientes
- `return_proba`: Incluir probabilidades (default: False)
- `include_input`: Incluir colunas originais (default: True)

**Método `predict_from_file()`:**
- `file_path`: Caminho para arquivo CSV
- `return_proba`: Incluir probabilidades (default: False)
- `include_input`: Incluir colunas originais (default: True)

### Validação Automática

A pipeline aplica automaticamente:

1. **Validação de schema**: Verifica se todas as features estão presentes e válidas
2. **Modo inference**: Garante que o target não está presente nos dados
3. **Transformação**: Aplica os mesmos mapeamentos binários e casting categórico do treinamento
4. **Preparação**: Remove colunas desnecessárias (ID) antes da predição

### Formato de Saída

**Sem probabilidades (`return_proba=False`):**
```python
   id  Age  Sex  ...  prediction
0   1   55    1  ...    Presence
1   2   62    0  ...    Absence
```

**Com probabilidades (`return_proba=True`):**
```python
   id  Age  Sex  ...  prediction  probability_Absence  probability_Presence
0   1   55    1  ...    Presence              0.35                   0.65
1   2   62    0  ...    Absence               0.72                   0.28
```

### Tratamento de Erros

```python
from heart_disease.pipelines.predict import PredictionPipeline

pipeline = PredictionPipeline()

try:
    # Erro: modelo não carregado
    results = pipeline.predict(data)
except ValueError as e:
    print(f"Erro: {e}")
    pipeline.load_model()  # Carregar antes

try:
    # Erro: dados contêm target
    results = pipeline.predict_from_file('data/with_target.csv')
except ValueError as e:
    print(f"Erro: {e}")

try:
    # Erro: modelo não existe
    pipeline = PredictionPipeline(model_alias='nonexistent')
    pipeline.load_model()
except ValueError as e:
    print(f"Erro: {e}")
```

### Reutilização de Componentes

A `PredictionPipeline` reutiliza os mesmos componentes do treinamento:

- **`DataLoader`**: Carregamento de CSV
- **`DataValidator`**: Validação em modo inference
- **`DataTransformer`**: Transformação de features
- **MLflow**: Carregamento do modelo ativo

Isso garante **consistência** entre treinamento e inferência.

## Testes

A estrutura de classe facilita o teste unitário de cada componente:

```python
import pytest
from heart_disease.pipelines.train import TrainingPipeline

def test_load_and_validate_data():
    pipeline = TrainingPipeline()
    df = pipeline._load_and_validate_data()
    assert len(df) > 0
    assert "Heart Disease" in df.columns

def test_prepare_features():
    pipeline = TrainingPipeline()
    df = pipeline._load_and_validate_data()
    X, y = pipeline._prepare_features(df)
    assert X.shape[0] == y.shape[0]
    assert "id" not in X.columns

def test_full_pipeline():
    pipeline = TrainingPipeline(n_iter=10, cv_folds=3)
    results = pipeline.run()
    assert "run_id" in results
    assert "metrics" in results
    assert pipeline.model_ is not None
```
