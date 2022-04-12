#!/bin/sh

jupyter nbconvert --to html notebooks/*.ipynb && mv notebooks/*.html data/notebooks_export/
