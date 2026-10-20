FROM intel4coro/yxzhan-2dcram-2dvrb-2dlab-eb909a@sha256:5190ba66716b6e911aeac881d5f5a4aa6afb3c5c7c1029c4e8171777381f5cd3
USER root
ENV ACCEPT_EULA=YES PRIVACY_CONSENT=YES OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_DRIVER_CAPABILITIES=all PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

# Reuse the same pinned CRAM/ROS/Isaac Sim installation as the verified demo.
RUN mkdir -p /workspace/third_party && \
    git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /workspace/third_party/IsaacLab && \
    test "$(git -C /workspace/third_party/IsaacLab rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0
COPY rl/install.sh rl/build-constraints.txt rl/runtime-constraints.txt /workspace/rl/
RUN bash /workspace/rl/install.sh

# Make the original assets available at the workspace paths used during training.
# Changes below belong only to this derived image, not the original Lab image.
RUN ln -s /home/jovyan/cram-vrb-lab /workspace/cram-vrb-lab
COPY runtime/cram_vrb_lab/ /home/jovyan/cram-vrb-lab/cram_vrb_lab/
COPY runtime/giskard_server.py /home/jovyan/cram-vrb-lab/demos/giskard_server.py
COPY . /workspace/
RUN python3 /workspace/runtime/unpack.py && \
    /home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python /workspace/integration/prepare_full_scene.py && \
    rm -f /home/jovyan/.config/autostart/GARMI-Apartment-autostart.desktop && \
    mkdir -p /workspace/results/cram-rl-web && \
    chown -R jovyan:users /workspace && \
    chmod +x /workspace/binder/entrypoint.sh

ENV CODE_WORKING_DIRECTORY=/workspace
USER jovyan
EXPOSE 8888
ENTRYPOINT ["/workspace/binder/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
