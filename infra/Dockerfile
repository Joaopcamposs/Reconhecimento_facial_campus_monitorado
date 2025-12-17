# pull official base image
FROM python:3.11-slim

# set work directory
WORKDIR /reconhecimento_facial

# copiar projeto
COPY ../src/infra .

# instalar dependencias
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3-lxml  \
    python3-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# configurar variáveis de ambiente de linguagem e horario
ENV LANG pt_BR.UTF-8
ENV LANGUAGE pt_BR:pt
ENV LC_ALL pt_BR.UTF-8

# instalar dependencias do python
RUN pip install --upgrade pip
RUN pip install uv
RUN uv sync

# Criar diretório de imagens
RUN mkdir -p /reconhecimento_facial/pictures

# Adiciona o .venv/bin ao PATH do container
ENV PATH="/reconhecimento_facial/.venv/bin:$PATH"
ENV PYTHONPATH=/reconhecimento_facial

# expose the 8004 port from the localhost system
EXPOSE 8004

# run app
CMD ["uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]
