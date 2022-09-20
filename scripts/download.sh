#!/bin/sh
scp sitnarf@strf.cz:~/projects/homage-fl/Pipfile.lock /home/sitnarf/projects/homage-fl
scp sitnarf@strf.cz:~/projects/homage-fl/Pipfile /home/sitnarf/projects/homage-fl
pipenv install --skip-lock
