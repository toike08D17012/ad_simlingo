FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    locales \
    build-essential \
    tzdata \
    clang \
    curl \
    wget \
    git \
    unzip \
    cmake \
    libffi-dev \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libgl1-mesa-dev \
    libopencv-dev \
    libsdl2-2.0 \
    xserver-xorg \  
    libvulkan1 \
    libomp5 \
    xdg-user-dirs \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    openjdk-8-jdk \
    libxml-parser-perl \
    && rm -rf /var/lib/apt/lists/*libpng-dev libtiff5-dev libjpeg-dev

ENV TZ=Asia/Tokyo

ARG WORKSPACE=/WORKSPACE

ARG USER_UID
ARG USER_GID
ARG USER_NAME
ARG GROUP_NAME=${USER_NAME}

ENV HOME=/home/${USER_NAME}

RUN groupadd --gid ${USER_GID} ${GROUP_NAME} \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} --groups sudo --base-dir /home --shell /bin/bash ${USER_NAME} \
    && touch ${HOME}/.sudo_as_admin_successful \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo '${USER_NAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} /home

### install pyenv
USER ${USER_NAME}
ENV PYTHON_VERSION=3.8.18 \
    UV_PROJECT_ENVIRONMENT=${HOME}/.venv

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}:/root/.cargo/bin:${HOME}/.local/bin:${PATH}"

ENV WORK_DIR=${HOME}/workspace/simlingo

ENV UV_PROJECT=${WORK_DIR}/docker/python

# pyproject.toml と uv.lock のコピー
COPY python/pyproject.toml python/uv.lock ${UV_PROJECT}/

# 仮想環境作成（指定バージョンを使用）
RUN uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${UV_PROJECT_ENVIRONMENT} \
    && uv sync \ 
    && python -c "import language_evaluation; language_evaluation.download('coco')"

# RUN set -ex \
#     && curl https://pyenv.run | bash \
#     && pyenv update \
#     && pyenv install ${PYTHON_VERSION} \
#     && pyenv global ${PYTHON_VERSION} \
#         && pyenv rehash

### install python library
# RUN pip install wheel==0.41.2 packaging

# COPY requirements.txt .
# RUN python -m pip install --upgrade pip \
#     && pip install -r requirements.txt

# RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# RUN pip install flash-attn==2.7.0.post2

### carla
USER ${USER_NAME}

ENV CARLA_ROOT=${HOME}/software/carla0915
WORKDIR ${CARLA_ROOT}

ENV SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
ENV LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
ENV PYTHONPATH="${WORK_DIR}:${CARLA_ROOT}/PYTHONAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}"
