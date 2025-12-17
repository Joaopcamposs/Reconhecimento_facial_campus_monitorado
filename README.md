# Reconhecimento Facial - Campus Monitorado

**Trabalho Científico e TCC do curso Engenharia de Computação**

API para reconhecimento facial desenvolvida com Python, FastAPI, MySQL e Docker.

**Objetivo:** Controlar câmeras IP no campus IFTM CAUPT, aplicando reconhecimento facial.

- **Autor:** João Pedro Vieira Campos
- **Orientador:** Ernani Viriato de Melo

## Requisitos

- Python 3.11+
- Docker e Docker Compose (para banco de dados)
- Webcam (para uso local) ou câmeras IP

## Instalação

### 1. Instalar dependências

```bash
# Usando uv (recomendado)
uv sync

# Ou usando pip
pip install -e .

# Para testes
uv sync --extra test
```

### 2. Iniciar o banco de dados

```bash
# Iniciar apenas o MySQL
docker-compose up -d reconhecimento_facial_db

# Aguardar o banco estar pronto
docker-compose logs -f reconhecimento_facial_db
```

### 3. Iniciar a API

```bash
# Modo desenvolvimento (com acesso à webcam)
uvicorn main:app --host 0.0.0.0 --port 8004 --reload

# Ou usando Docker completo (sem webcam)
docker-compose up -d --build
```

## Uso

### Acessar documentação (Swagger)
http://localhost:8004/docs

### Fluxo de funcionamento (RECOMENDADO)

1. **Verificar status do sistema**
   ```
   GET /status
   ```

2. **Capturar fotos com modo automático** (RECOMENDADO)
   
   Abra no navegador para ver o vídeo em tempo real. As fotos são capturadas automaticamente quando um rosto é detectado:
   ```
   GET /captura-auto/webcam/{nome_pessoa}?quantidade=20&intervalo=0.5
   ```
   
   Parâmetros opcionais:
   - `quantidade`: número de fotos (padrão: 20)
   - `intervalo`: segundos entre capturas (padrão: 0.5)

3. **Treinar o algoritmo**
   ```
   GET /treinamento
   ```

4. **Realizar reconhecimento facial em tempo real**
   
   Abra no navegador:
   ```
   GET /video/webcam
   ```

### Análise de arquivos de vídeo

Para analisar vídeos gravados (sem depender de stream ao vivo):

1. **Upload do vídeo**
   ```
   POST /video/upload
   (form-data: file=video.mp4)
   ```

2. **Analisar com visualização** (abra no navegador)
   ```
   GET /video/analisar-arquivo?caminho=/path/to/video.mp4
   ```

3. **Analisar e obter JSON**
   ```
   POST /video/analisar-arquivo-json?caminho=/path/to/video.mp4
   ```

4. **Listar vídeos disponíveis**
   ```
   GET /videos
   ```

### Endpoints de gerenciamento

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/status` | GET | Status do sistema |
| `/fotos` | GET | Listar fotos capturadas |
| `/fotos` | DELETE | Deletar todas as fotos |
| `/pessoas` | GET | Listar pessoas cadastradas |
| `/cameras` | GET | Listar câmeras cadastradas |
| `/videos` | GET | Listar vídeos para análise |

### Modo manual (alternativo)

Se preferir controle manual sobre cada captura:
```
GET /fotos/webcam/{nome_pessoa}   # Inicia stream
POST /capturar                     # Captura uma foto (repetir 20x)
```

## Configuração

### Variáveis de ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `USE_WEBCAM_FALLBACK` | Usar webcam quando câmera IP não disponível | `true` |
| `DEFAULT_WEBCAM_INDEX` | Índice da webcam padrão | `0` |
| `IN_DOCKER` | Indica se está rodando em Docker | `false` |

### macOS (Apple Silicon)

O projeto está configurado para funcionar com macOS ARM (M1/M2/M3) usando o backend AVFoundation para acesso à câmera.

## Testes

```bash
# Instalar dependências de teste
uv sync --extra test

# Rodar testes
pytest
```

## Arquitetura

```
├── api.py              # Endpoints da API
├── config.py           # Configurações e paths
├── crud.py             # Operações de banco de dados
├── database.py         # Conexão com banco de dados
├── facial_recognition.py # Reconhecimento facial
├── models.py           # Modelos SQLAlchemy
├── pictures_capture.py # Captura de fotos
├── schema.py           # Schemas Pydantic
├── training.py         # Treinamento do modelo
├── pictures/           # Diretório de fotos capturadas
└── tests/              # Testes automatizados
```

## Observações

- Para capturar fotos, a pessoa deve estar em frente à câmera
- São necessárias pelo menos algumas fotos para treinar (recomendado: 20 por pessoa)
- O reconhecimento facial requer que o modelo tenha sido treinado previamente
- Em Docker, a webcam não está disponível - use câmeras IP ou rode localmente