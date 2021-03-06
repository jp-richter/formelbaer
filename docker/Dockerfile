FROM s876cnsm:5000/pfahler-cuda
SHELL ["/bin/bash", "-c"]
#ARG USER="schill"
#ARG USERID=8216
ARG USER="richter2"
ARG USERID=8289
ARG GROUPID=9004
ARG GROUPNAME="s876a1"
# Switch to root, otherwise the permissions to create a new user are missing.
USER "root" 

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get install -y \
	curl \
	bzip2 \
	psmisc \
	nano \
	git \
	openssh-client \
	texlive \
	texlive-latex-extra \
	imagemagick
RUN sed 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml > policy.xml && mv policy.xml /etc/ImageMagick-6/

ENV MINICONDA_VERSION 4.7.10
ENV MINICONDA_MD5 1c945f2b3335c7b2b15130b1b2dc5cf4

# RUN curl -s https://repo.continuum.io/archive/Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh -o anaconda.sh && \
# 	echo "${ANACONDA_MD5}  anaconda.sh" > anaconda.md5 && \
# 	if [ $(md5sum -c anaconda.md5 | awk '{print $2}') != "OK" ] ; then exit 1; fi && \
# 	mkdir -p /opt && \
# 	sh ./anaconda.sh -b -p /opt/conda && \
# 	rm anaconda.sh anaconda.md5

RUN curl -s https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -o anaconda.sh && \
	echo "${MINICONDA_MD5}  anaconda.sh" > anaconda.md5 && \
	if [ $(md5sum -c anaconda.md5 | awk '{print $2}') != "OK" ] ; then exit 1; fi && \
	mkdir -p /opt && \
	sh ./anaconda.sh -b -p /opt/conda && \
	rm anaconda.sh anaconda.md5

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

COPY environment.yml .
RUN source $CONDA_DIR/bin/activate && conda env create -f environment.yml && rm environment.yml

# Create a new user.
RUN mkdir -p "/home/${USER}" && \
	echo "${USER}:x:${USERID}:${GROUPID}:${USER}:/home/${USER}:/bin/bash" >> /etc/passwd && \
	echo "${GROUPNAME}:x:${GROUPID}:${USER}" >> /etc/group                               && \
	chown -R "${USERID}:${GROUPID}" "/home/${USER}"

WORKDIR "/home/${USER}"
RUN mkdir arxiv
COPY . arxiv/
RUN chown -R "${USER}" arxiv
USER "${USER}"
RUN echo "source $CONDA_DIR/bin/activate" >> ~/.bashrc
RUN echo "conda activate arxiv" >> ~/.bashrc
WORKDIR "/home/${USER}/arxiv"
CMD ["bash"]
