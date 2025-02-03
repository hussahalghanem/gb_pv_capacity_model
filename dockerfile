FROM python:3.10

# Set working directory to /gb_pv_capacity_model
WORKDIR /gb_pv_capacity_model

# Copy requirements files
COPY requirements.txt /gb_pv_capacity_model/requirements.txt
COPY requirements_dev.txt /gb_pv_capacity_model/requirements_dev.txt

# Install production requirements
RUN pip install --no-cache-dir -r /gb_pv_capacity_model/requirements.txt > /dev/null

# Install development requirements if buildmode is set to dev
ARG buildmode=prod
RUN if [ "$buildmode" = "dev" ]; then \
         pip install --no-cache-dir -r /gb_pv_capacity_model/requirements_dev.txt > /dev/null ;\
       fi

# Copy application code
# COPY . /gb_pv_capacity_model/

# Expose port for JupyterLab
EXPOSE 5000

# Run JupyterLab
CMD ["jupyter", "lab", "--allow-root", "--ip", "0.0.0.0", "--port", "5000", "--no-browser", "--notebook-dir=/gb_pv_capacity_model"]
