



# FROM continuumio/miniconda3


# # Make RUN commands use `bash --login`:
# SHELL ["/bin/bash", "--login", "-c"]


# load default mlo image
FROM mlo_base

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN sudo -s


RUN sudo conda create --name MyProject
RUN source activate MyProject

COPY ./base.yml .
RUN sudo conda env update -f base.yml


RUN source activate MyProject
COPY ./ml.yml .
RUN sudo conda env update -f ml.yml -n MyProject

RUN sudo conda install -c conda-forge wandb -n MyProject

RUN pip install -U "ray[tune]"
RUN pip install -e .

RUN git clone git@github.com:RECOVERcoalition/Reservoir.git
RUN cd Reservoir/
RUN python setup.py develop
RUN cd ../

RUN exit
WORKDIR /home/hokarami

COPY ./data ./data
ADD ./hello.py ./codes/



# WORKDIR /codes

# COPY ./codes .
# COPY c\\DATA\\data\\processed\\p12_full_seft               ./data/p12_full_seft
# COPY c\\DATA\\data\\processed\\physio2019_1d_HP_std_AB     ./data/physio2019_1d_HP_std_AB



# CMD ["activate", "MyProject" ]

# CMD [ "python", "./codes/THP_new/Main.py" ]


# docker build . -t p22 --progress=plain
# docker run -it MyProject



# docker rm  $(docker ps -q -a)
# docker rmi $(docker images -f "dangling=true" -q)


# docker run -it --name test --mount  source=p22_data,target=/codes/ MyProject 
# docker run -it --name test -v  p22_data:/codes/ MyProject 


# running containers:
# docker ps -a

# remove all containers:
# docker rm  $(docker ps -q -a)




