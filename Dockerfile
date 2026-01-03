FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /robustmanipulators
WORKDIR /robustmanipulators

# Core system packages
RUN apt update && apt install -y git curl

# Install Robust Gym
RUN git clone https://github.com/SafeRL-Lab/Robust-Gymnasium.git

ENV TORCH_CUDA_ARCH_LIST=Turing

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && . $HOME/.local/bin/env \
    && uv venv --python 3.11 --prompt 🦾 venv \
    && . venv/bin/activate \
    && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install tyro \
    && uv pip install jax[cuda12] \
    && uv pip install tensorboard \
    && cd Robust-Gymnasium \
    && uv pip install -r requirements.txt  \
    && uv pip install -e .

RUN . $HOME/.local/bin/env \
    && . venv/bin/activate \
    && apt install -y ninja-build gettext cmake unzip curl  \
    && git clone --single-branch --depth=1 https://github.com/neovim/neovim \
    && cd neovim \
    && make CMAKE_BUILD_TYPE=Release \
    && make install \
    && ln -s /usr/local/bin/nvim /usr/bin/nvim \
    && . $HOME/.local/bin/env \
    && uv pip install pynvim \
    && sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'

# RUN cd Robust-Gymnasium && \
    
#     uv pip install -e .

#ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        ffmpeg \
        freeglut3-dev \
        swig \
        xvfb \
        vim \
        libxrandr2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#ENV MUJOCO_GL="egl"
#ENV PYOPENGL_PLATFORM="egl"


#RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' >> /usr/share/glvnd/egl_vendor.d/10_nvidia.json

#USER $USERNAME
#RUN mkdir -p /home/vscode/robotic_manipulation_robust
#WORKDIR /home/vscode/robotic_manipulation_robust

COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh
ENTRYPOINT ["/root/entrypoint.sh"]

# Bashrc
RUN echo "export PS1=$''" >> ~/.bashrc \
 #&& echo "alias vim='/usr/bin/nvim'" >> ~/.bashrc \ 
 #&& echo "alias diff='diff --color --palette=':ad=36:de=31:ln=33''" >> ~/.bashrc \
 && echo "alias pip='uv pip'" >> ~/.bashrc \
 && echo ". /robustmanipulators/venv/bin/activate" >> ~/.bashrc \
 #&& echo "cd /puffertank/pufferlib" >> ~/.bashrc
RUN apt clean
CMD ["/bin/bash"]
