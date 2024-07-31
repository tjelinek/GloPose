# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment YAML file to the working directory
COPY environment.yml .

# Create a new Conda environment based on the environment YAML file
RUN conda env create -n FlowTracker
RUN conda install -y nvidia/label/cuda-12.4.0::cuda

# Make sure the environment is activated:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install additional packages if needed
RUN conda install -n myenv numpy pandas scikit-learn

# Set the default command to activate the environment and run a shell
CMD ["conda", "run", "-n", "myenv", "/bin/bash"]
