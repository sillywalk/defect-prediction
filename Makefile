TEST_PATH=./

all: test git 

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	@- test -d venv || virtualenv venv
	@- . venv/bin/activate; pip install -Ur requirements.txt
	@- touch venv/bin/activate

test: venv 
	@echo "Running unit tests."
	@echo ""
	@- . venv/bin/activate; nosetests -s $(TEST_PATH); deactivate
	@echo ""

clean:
	@echo "Cleaning *.pyc, *.DS_Store, and other junk files..."
	@- find . -name '*.pyc' -exec rm -f {} +
	@- find . -name '*.pyo' -exec rm -f {} +
	@- find . -name '__pycache__' -exec rm -rf {} +
	@echo ""

doc: venv
	@- . venv/bin/activate && cd docs; make html; cd .. 

git: doc clean
	@echo "Syncing with repository"
	@echo ""
	@- git add --all .
	@- git commit -am "Autocommit from makefile"
	@- git push origin master
	
