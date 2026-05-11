# FIAP Projeto Fase 4 — API de Previsão de Retorno de Ações

> API REST com modelo BiLSTM para previsão de retorno do próximo dia útil da ação AAPL, com monitoramento em produção via Prometheus, Grafana e MLflow.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Modelo](#modelo)
- [Endpoints](#endpoints)
- [Como Rodar Localmente](#como-rodar-localmente)
- [Deploy com Docker](#deploy-com-docker)
- [Monitoramento](#monitoramento)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias](#tecnologias)

---

## Visão Geral

Este projeto implementa uma pipeline completa de Machine Learning para previsão de séries temporais financeiras:

- **Treinamento**: modelo BiLSTM treinado com dados históricos da AAPL (2002–2026)
- **API**: FastAPI com autenticação JWT, rota de predição e métricas
- **Monitoramento**: Prometheus + Grafana para métricas de infraestrutura, MLflow para rastreamento do modelo
- **Deploy**: containerizado com Docker, deployado no Railway

---

## Arquitetura

```
┌─────────────┐     POST /api/predict     ┌──────────────────┐
│   Cliente   │ ─────────────────────────▶│   FastAPI (API)  │
└─────────────┘                           └────────┬─────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          ▼                        ▼                        ▼
                  ┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
                  │  BiLSTM Model │     │   Prometheus     │     │     MLflow      │
                  │  (.keras)     │     │  (métricas API)  │     │ (drift modelo)  │
                  └───────────────┘     └────────┬─────────┘     └─────────────────┘
                                                 ▼
                                        ┌─────────────────┐
                                        │     Grafana     │
                                        │  (dashboards)   │
                                        └─────────────────┘
```

---

## Modelo

| Parâmetro | Valor |
|---|---|
| Arquitetura | BiLSTM 64→32 + Dense 16 |
| Framework | TensorFlow 2.21.0 |
| Janela temporal | 30 dias |
| Features | 13 (RETURN, RSI, MACD_HIST, volatilidade, momentum...) |
| Normalização | Z-score por janela |
| Loss | Huber |
| Target | Retorno do dia seguinte (`return_t+1`) |
| MAE | 1.23% |
| RMSE | 1.74% |
| Directional Accuracy | 53.2% (+1.6pp vs baseline) |

---

## Endpoints

### Autenticação

```
POST /api/auth/login
```
Retorna JWT token para uso nas rotas protegidas.

### Predição

```
POST /api/predict/
Authorization: Bearer <token>
```

**Body:**
```json
{
  "ticker": "AAPL",
  "history": [
    { "open": 182.1, "high": 184.5, "low": 181.0, "close": 183.2, "volume": 52000000 }
    // mínimo 56 registros OHLCV ordenados do mais antigo ao mais recente
  ]
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "predicted_return_pct": 0.1438,
  "direction": "UP",
  "confidence": "LOW",
  "model_version": "2.0.0"
}
```

### Métricas do Modelo

```
GET /api/model/metrics
Authorization: Bearer <token>
```

Retorna MAE, RMSE, Directional Accuracy e metadados do modelo carregado.

### Health Check

```
GET /api/health
```

### Métricas Prometheus

```
GET /metrics
```

---

## Como Rodar Localmente

### Pré-requisitos

- Python 3.10+
- pip

### Instalação

```bash
git clone https://github.com/seu-usuario/FIAP_projeto_fase_4.git
cd FIAP_projeto_fase_4

pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um `.env` na raiz:

```env
MODEL_ARTIFACTS_PATH=src/model_artifacts
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Iniciar a API

```bash
uvicorn run:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

---

## Deploy com Docker

### Stack completa (API + Prometheus + Grafana + MLflow)

```bash
docker compose up -d
```

| Serviço | URL |
|---|---|
| API | http://localhost:8000 |
| Documentação | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5001 |

### Apenas a API

```bash
docker build -t fiap-api .
docker run -p 8000:8000 fiap-api
```

---

## Monitoramento

### Prometheus + Grafana

O projeto inclui um dashboard Grafana pré-configurado (`grafana_dashboard.json`) com:

- Status do modelo (ONLINE/OFFLINE)
- Total de requisições e taxa req/s
- Latência média, P50, P95, P99 por endpoint
- Uso de CPU e memória do processo
- Predições por direção (UP/DOWN/NEUTRAL)
- Latência de inferência do modelo
- Distribuição dos retornos previstos

**Importar dashboard:**
1. Acesse `http://localhost:3000`
2. Dashboards → Import → Upload JSON
3. Selecione `grafana_dashboard.json`

### MLflow

Rastreamento de predições em produção com detecção automática de drift:

```
http://localhost:5001
```

Experimento: `production_AAPL` — loga métricas a cada 50 predições ou 5 minutos, com alerta automático quando o viés de direção supera 30pp.

---

## Estrutura do Projeto

```
FIAP_projeto_fase_4/
├── run.py                          # entrypoint da aplicação
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── prometheus.yml                  
├── grafana/
│   └── provisioning/
│       └── datasources/
│           └── prometheus.yml
├── src/
│   ├── api/
│   │   └── app.py                 
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── predict_routes.py
│   │   ├── metrics_routes.py
│   │   ├── health_routes.py
│   │   └── loader_model.py
│   ├── services/
│   │   ├── model_loader_services.py
│   │   ├── predict_service.py
│   │   └── auth_services.py
│   ├── models/
│   │   └── predict_models.py
│   ├── monitoring/
│   │   ├── metrics_middleware.py   # Prometheus middleware
│   │   └── model_monitor.py        # MLflow monitor
│   ├── model_artifacts/
│   │   ├── model.keras
│   │   └── inference_metadata.json
│   └── log/
│       └── logs.py
└── notebooks/
    └── train_models.ipynb
```

---

## Tecnologias

| Camada | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Modelo | TensorFlow 2.21 / BiLSTM |
| Autenticação | JWT |
| Monitoramento | Prometheus + Grafana |
| Rastreamento ML | MLflow |
| Containerização | Docker + Docker Compose |
| Deploy | Railway |
| Linguagem | Python 3.10 |