#!/bin/bash
nvidia-docker build -t freda/pytorch .
nvidia-docker run --rm -ti freda/pytorch:latest /bin/bash
