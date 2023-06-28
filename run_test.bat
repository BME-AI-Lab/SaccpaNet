python -m coverage run --branch --source src -m pytest tests/ 
python -m coverage xml
python -m coverage report -m