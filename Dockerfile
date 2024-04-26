FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app
COPY . /app

ENV HF_HOME=/app/models

RUN pip --no-cache-dir install -r requirements.txt
RUN python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

EXPOSE 8888

CMD ["jupyter", "notebook", "./src", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''"]