## Image for building MCU code including other tooling
##
FROM debian:bookworm as tools

WORKDIR /opt
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
     && apt install -y \
     sudo \
     virtualenv python3-dev libffi-dev build-essential pkg-config\
     libncurses-dev \
     avrdude gcc-avr binutils-avr avr-libc \
     stm32flash dfu-util libnewlib-arm-none-eabi \
     gcc-arm-none-eabi binutils-arm-none-eabi libusb-1.0-0 libusb-1.0-0-dev \
     python3-numpy python3-matplotlib \
     && apt clean

COPY --from=mkuf/klipper:latest /opt/klipper ./klipper
COPY --from=mkuf/klipper:latest /opt/venv ./venv
###

## Build Simulavr and klipper.elf
## For manual builds, not uploaded to the docker registry
FROM tools as build-simulavr

RUN apt update \
     && apt install -y \
     git cmake swig \
     python3-dev \
     && apt clean

WORKDIR /opt/klipper
COPY config.simulavr .config
RUN make 

WORKDIR /opt
RUN mkdir run
RUN groupadd simulavr --gid 1000 \
     && useradd simulavr --uid 1000 --gid simulavr \
     && usermod simulavr --append --groups dialout

RUN git clone git://git.savannah.nongnu.org/simulavr.git \
     && cd simulavr \
     && git checkout release-1.1.0 \
     && make python \
     && make build \
     && chown -R simulavr:simulavr /opt/simulavr

USER simulavr
ENV PYTHONPATH=/opt/simulavr/build/pysimulavr/
VOLUME ["/opt/printer_data/run"]
ENTRYPOINT ["klipper/scripts/avrsim.py"]
CMD ["-p", "/opt/printer_data/run/simulavr.tty", "/opt/klipper/out/klipper.elf"]
