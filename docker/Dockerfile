# Use an official conda image from continuumio
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file to the container
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set the environment variables
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install any additional packages using pip
RUN pip install --no-cache-dir some-package

# Copy the rest of the application code to the container
COPY . .

# Set the entry point to use the conda environment
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "your_script.py"]
