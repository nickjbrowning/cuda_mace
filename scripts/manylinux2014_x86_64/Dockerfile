FROM quay.io/pypa/manylinux2014_x86_64

# Set environment variable for CUDA version
ARG CUDA_VER="12-4"

# Install system dependencies and CUDA
RUN yum install -y yum-utils gcc gcc-c++ make zlib-devel bzip2-devel libffi-devel \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && yum install -y \
    cuda-toolkit-${CUDA_VER} \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

    # Remove all other Python versions
ENV PATH="/opt/python/cp311-cp311/bin:${PATH}"

RUN ln -sf /opt/python/cp311-cp311/bin/python3.11 /opt/python/cp311-cp311/bin/python

RUN python -m ensurepip --upgrade

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

RUN yum install -y git-all

# Verify the CUDA installation
RUN echo "CUDA_HOME: ${CUDA_HOME}" && \
    nvcc --version

RUN pip install torch==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install numpy cmake

RUN mkdir /workspace

# Add the remove_python.sh script to the container
COPY remove_unused_python.sh /scripts/remove_unused_python.sh
# Make the script executable
RUN chmod +x /scripts/remove_unused_python.sh
# Run the script to remove all Python versions except Python 3.11
RUN /scripts/remove_unused_python.sh python3.11

# Clean up to reduce image size
#RUN yum clean all && rm -rf /var/cache/yum/*

# Default command (bash)
CMD ["/bin/bash"]
