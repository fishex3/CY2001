# R >= 4.3 required by ConnectednessApproach (CRAN); Debian bookworm default R is 4.2.x.
FROM rocker/r-ver:4.4.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    R_HOME=/usr/local/lib/R

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-turbo8-dev \
    libgit2-dev \
    libssh2-1-dev \
    libpq-dev \
    libgmp-dev \
    libmpfr-dev \
    libnlopt-dev \
    libudunits2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN Rscript -e "options(Ncpus = max(1L, parallel::detectCores())); install.packages( \
    c( \
      'ConnectednessApproach', \
      'ggplot2', \
      'lubridate', \
      'dplyr', \
      'tidyr', \
      'zoo', \
      'xts' \
    ), \
    repos = 'https://cloud.r-project.org', \
    dependencies = TRUE \
  )" \
  && Rscript -e "stopifnot(getRversion() >= numeric_version('4.3')); library(ConnectednessApproach)"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
