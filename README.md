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
