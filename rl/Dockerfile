FROM intel4coro/yxzhan-2dcram-2dvrb-2dlab-eb909a@sha256:5190ba66716b6e911aeac881d5f5a4aa6afb3c5c7c1029c4e8171777381f5cd3
USER root
ENV ACCEPT_EULA=YES PRIVACY_CONSENT=YES OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
WORKDIR /workspace
# Source remains mounted at /workspace so changes are available without rebuilding.
# The install script is also used for the persistent development container.
ENTRYPOINT ["/usr/bin/sleep"]
CMD ["infinity"]
