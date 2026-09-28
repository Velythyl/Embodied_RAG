# Embodied-RAG (original release) semantic forest builder + RAGMAP adapter, as an
# "area_mapping" container. CPU only: the build is agglomerative clustering plus
# LLM calls to an OpenAI-compatible server (RAGMAP's VLM service), reached over
# the host network. Upstream code is copied unmodified; see ragmap_adapter/.
FROM python:3.11-slim

ARG EMBODIED_RAG_GIT_SHA=unknown
ENV EMBODIED_RAG_ROOT=/opt/Embodied_RAG \
    EMBODIED_RAG_GIT_SHA=${EMBODIED_RAG_GIT_SHA} \
    PYTHONPATH=/opt/Embodied_RAG \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/Embodied_RAG
COPY ragmap_adapter/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /opt/Embodied_RAG
RUN printf '#!/bin/sh\nexec python -m ragmap_adapter.run "$@"\n' > /usr/local/bin/ragmap-run \
    && chmod +x /usr/local/bin/ragmap-run \
    && python -c "import generate_semantic_forest; \
from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor; \
from embodied_nav.llm import LLMInterface; from ragmap_adapter.run import main; print('imports ok')" \
    && ragmap-run --help >/dev/null

WORKDIR /work
ENTRYPOINT ["ragmap-run"]
CMD ["--help"]
