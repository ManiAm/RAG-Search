FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    nano

RUN python3.10 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install sentence-transformers
RUN pip install fastapi uvicorn

WORKDIR /app
COPY . .

EXPOSE 5005

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5005"]
