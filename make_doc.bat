sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/api src/lib
sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/api src/models
sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/api src/helpers
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 01-Random_Search/codes
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 02-Manual_Search
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 03-Pretraining
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 04-Weight_Transfer
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 05-Finetuning
@REM sphinx-apidoc -e -M --implicit-namespaces  -f -o docs/source/experiments 06-Posture_Classification/codes
./docs/make.bat html