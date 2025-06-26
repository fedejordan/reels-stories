#!/bin/bash

id=$1

python main.py "$id" video && python generar-descripcion-tiktok.py "$id"
