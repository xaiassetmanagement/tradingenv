# To build:
# $ sudo docker build -f Dockerfile -t tradingenv .
FROM python:3.12
WORKDIR /home
COPY . .
RUN pip install .
RUN python -c "import tradingenv;print(tradingenv.__version__)"
