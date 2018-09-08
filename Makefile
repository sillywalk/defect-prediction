TEST_PATH=./
VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3

all: test clean git

venv: 
	$(VENV_PATH)/bin/activate


test:
	@echo "Running unit tests."
	@echo ""
	@nosetests -s $(TEST_PATH)
	@echo ""

clean:
	@echo "Cleaning *.pyc, *.DS_Store, and other junk files..."
	@- find . -name '*.pyc' -exec rm -f {} +
	@- find . -name '*.pyo' -exec rm -f {} +
	@- find . -name '__pycache__' -exec rm -rf {} +
	@echo ""

git: clean
	@echo "Syncing with repository"
	@echo ""
	@- git add --all .
	@- git commit -am "Autocommit from makefile"
	@- git push origin master
